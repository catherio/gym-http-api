extern crate serde_json;
extern crate curl;
extern crate rand;

use std::collections::BTreeMap;

use serde_json::Value;
use serde_json::value::{ToJson, from_value};
use serde_json::ser::to_string_pretty;

use curl::easy::{Easy, List};
use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub enum Space {
	DISCRETE{n: u64},
	BOX{shape: Vec<u64>, high: Vec<f64>, low: Vec<f64>},
	TUPLE{spaces: Vec<Box<Space>>}
}

impl Space {
	fn from_json(info: &Value) -> Space {
		match info.find("name").unwrap().as_str().unwrap() {
			"Discrete" => {
				let n = info.find("n").unwrap().as_u64().unwrap();
				Space::DISCRETE{n: n}
			},
			"Box" => {
				let mut shape = Vec::new();
				for val in info.find("shape").unwrap().as_array().unwrap() {
					shape.push(val.as_u64().unwrap());
				}
				let mut high = Vec::new();
				for val in info.find("high").unwrap().as_array().unwrap() {
					high.push(val.as_f64().unwrap());
				}
				let mut low = Vec::new();
				for val in info.find("low").unwrap().as_array().unwrap() {
					low.push(val.as_f64().unwrap());
				}
				Space::BOX{shape: shape, high: high, low: low}
			},
			"Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
			e @ _ => panic!("Unrecognized space name: {}", e)
		}
	}
	pub fn sample(&self) -> Vec<f64> {
		let mut rng = thread_rng();
		match *self {
			Space::DISCRETE{n} => {
				vec![(rng.gen::<u64>()%n) as f64]
			},
			Space::BOX{ref shape, ref high, ref low} => {
				let mut ret = Vec::new();
				let mut index = 0;
				for &i in shape {
					for _ in 0..i {
						ret.push(rng.gen_range(low[index], high[index]));
						index += 1;
					}
				}
				ret
			},
			Space::TUPLE{ref spaces} => {
				let mut ret = Vec::new();
				for space in spaces {
					ret.extend(space.sample());
				}
				ret
			}
		}
	}
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct State {
	pub observation:	Vec<f64>,
	pub reward:			f64,
	pub done:			bool,
	pub info:			Value
}

#[allow(dead_code)]
pub struct Environment {
	client:			Box<Client>,
	instance_id:	String,
	act_space:		Space,
	obs_space:		Space
}

impl Environment {
	pub fn action_space(&self) -> Space {
		self.act_space.clone()
	}
	pub fn observation_space(&self) -> Space {
		self.obs_space.clone()
	}
	pub fn reset(&mut self) -> Vec<f64> {
		let observation = self.client.post("/v1/envs/".to_string() + &self.instance_id + "/reset/", 
										   Value::Null);
		let mut ret = Vec::new();
		for val in observation.find("observation").unwrap().as_array().unwrap() {
			ret.push(val.as_f64().unwrap());
		}
		ret
	}
	pub fn step(&mut self, action: Vec<f64>, render: bool) -> State {
		let mut req = BTreeMap::new();
		req.insert("render", Value::Bool(render));
		match self.act_space {
			Space::DISCRETE{..} => {
				assert_eq!(action.len(), 1);
				req.insert("action", Value::U64(action[0] as u64));
			},
			Space::BOX{ref shape, ..} => {
				assert_eq!(action.len(), shape[0] as usize);
				req.insert("action", action.to_json());
			},
			Space::TUPLE{..} => panic!("Actions for Tuple spaces not implemented yet")
		}

		let state = self.client.post("/v1/envs/".to_string() + &self.instance_id + "/step/",
									 req.to_json());

		State {
			observation: from_value(state.find("observation").unwrap().clone()).unwrap(),
			reward: state.find("reward").unwrap().as_f64().unwrap(),
			done: state.find("done").unwrap().as_bool().unwrap(),
			info: state.find("info").unwrap().clone()
		}
	}
	pub fn monitor_start(&mut self, directory: String, force: bool, resume: bool) {
		let mut req = BTreeMap::new();
		req.insert("directory", Value::String(directory));
		req.insert("force", Value::Bool(force));
		req.insert("resume", Value::Bool(resume));
		self.client.post("/v1/envs/".to_string() + &self.instance_id + "/monitor/start/",
						 req.to_json());
	}
	pub fn monitor_stop(&mut self) {
		self.client.post("/v1/envs/".to_string() + &self.instance_id + "/monitor/close/",
						 Value::Null);
	}
}

pub struct Client {
	address:	String,
	handle:		Easy
}

impl Client {
    pub fn new(addr: String) -> Client {
    	let mut headers = List::new();
    	headers.append("Content-Type: application/json").unwrap();

    	let mut handle = Easy::new();
    	handle.http_headers(headers).unwrap();

    	Client{address: addr, handle: handle}
    }
    pub fn make(mut self, env_id: &str) -> Environment {
    	let mut req: BTreeMap<&str, &str> = BTreeMap::new();
    	req.insert("env_id", env_id);

    	let instance_id = self.post("/v1/envs/".to_string(), req.to_json());
    	let instance_id = match instance_id.find("instance_id") {
    		Some(id) => id.as_str().unwrap(),
    		None => panic!("Unrecognized environment id: {}", env_id)
    	};

    	let obs_space = self.get("/v1/envs/".to_string() + instance_id + "/observation_space/");
    	//println!("observation space json:\n{}", to_string_pretty(&obs_space).unwrap());

    	let act_space = self.get("/v1/envs/".to_string() + instance_id + "/action_space/");
    	//println!("action space json:\n{}", to_string_pretty(&act_space).unwrap());

    	Environment{client: Box::new(self), instance_id: instance_id.to_string(),
    				act_space: Space::from_json(act_space.find("info").unwrap()),
    				obs_space: Space::from_json(obs_space.find("info").unwrap())}
    }

    fn post(&mut self, route: String, request: Value) -> Value {
    	let request = to_string_pretty(&request).unwrap();
    	let data = request.as_bytes();
    	let url = self.address.clone() + &route;

    	self.handle.url(&url).unwrap();
	    self.handle.post(true).unwrap();
	    self.handle.post_field_size(data.len() as u64).unwrap();
	    self.handle.post_fields_copy(data).unwrap();
	    
	    let mut answer = Vec::new();
	    {
	    	let mut transfer = self.handle.transfer();
		    transfer.write_function(|data| {
		        answer.extend_from_slice(data);
		        Ok(data.len())
		    }).unwrap();
		    transfer.perform().unwrap();
	    }

	    serde_json::from_str(&String::from_utf8(answer).unwrap()).unwrap()
    }
    fn get(&mut self, route: String) -> Value {
    	let url = self.address.clone() + &route;

    	self.handle.url(&url).unwrap();
    	self.handle.post(false).unwrap();

    	let mut answer = Vec::new();
	    {
	    	let mut transfer = self.handle.transfer();
		    transfer.write_function(|data| {
		        answer.extend_from_slice(data);
		        Ok(data.len())
		    }).unwrap();
		    transfer.perform().unwrap();
	    }
	    
	    serde_json::from_str(&String::from_utf8(answer).unwrap()).unwrap()
    }
}