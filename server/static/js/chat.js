// cout <<"Welcome to Schedule-Bot. I can help schedule your workouts. Tell me a 3 hour period in which you plan to work out today."<<endl;
// string time;
// cin >> time; //in the form of 6-9 means 6am to 9am or 18-21 means 6pm to 9pm
// //will take string time and take first character turn into starting int and take 3rd character, turn into ending int
// cout<<"Please enter: your workout, time desired in minutes. Ask me for "schedule" when complete."<<endl;


// string userInput;
// int workoutTime [] =5;
// String workoutType [] =5;


// for (int i=0;i<5;i++){ //won't allow user to put in more than 5 workouts
// 	cin>>userInput;
// 	//take string up until comma and store in workoutType[i]
// 	//take number one space after comma and store in workoutTime[i]
// }

// //user a hardcoded database of workouts to aggregate which ones should go first in schedule
// bench press has highest score because it usually comes first
// upper body stuff like pushups and come next
// then bicep curls 
// then squats and legs etc

// match workoutType[i] with each possible prestored workout using ignorecase, if nothing matches, use math.random to aggregate a random score 
// and place accordingly into schedule based off intial string time received! 
// print out schedule in a text format!!!

// each index is a 15 minute block
var schedule = []

var responses = ["Anything else my dude?",
				 "Gotta hit some more zones, bro!",
				 "If you need suggestions, just ask!",
				 "Keep going, I believe in you!"]

var workouts = [
	"shoulderpress",
	"chestpress",
	"benchpress",
	"pushups",
	"curls",
	"squats",
	"plank"
]

var workoutMap = {
	"shoulderpress": 3,
	"chestpress": 5,
	"benchpress": 5,
	"pushups": 4,
	"curls": 3,
	"squats": 2,
	"plank": 1
}

var emojis = ["&#x1F605;", "&#x1F60B;", "&#x1F61C;"]

function removeLastLine() {
	$('#chatOut').children().last().remove();
}

function addMessage(msg, from) {
	var cls;
	if (from=="bot") {
		cls = "botText"
		from = "Schedule-Bot: "
	} else if (from=="None") {
		from = "&nbsp;&nbsp;&nbsp;&nbsp;"
		cls = "botText"
	} else {
		cls = "userText"
		from = "You: "
	}
	var message = "<p class='" + cls + "'>" + from + msg + "</p>"
	$("#chatOut").append(message);
}

function addResponse(input) {
	addMessage("Let me think... " + emojis[Math.floor(Math.random() * 3)], "bot")
	setTimeout(function(){
		removeLastLine()

		var matches = {}
		matches.shoulderpress = /shoulders? ?(press)?/ig.exec(input);
		matches.chestpress = /chests? ?(press)?/ig.exec(input);
		matches.benchpress = /bench ?(press)?/ig.exec(input);
		matches.pushups = /push ?(ups?)?/ig.exec(input);
		matches.curls = /(curls)|(biceps)|(swole)/ig.exec(input);
		matches.squats = /(squat)|(butt)|(curvy)|(legs)/ig.exec(input);
		matches.plank = /(abs)|(full ?body)|(swole)/ig.exec(input);
		matches.schedule = /(schedule)|(calender)|(time)|(workout)/ig.exec(input);
	
		var max_matches = 0
		var workout = workouts[Math.floor(Math.random()*workouts.length)]
		for (var key in matches) {
			// check if the property/key is defined in the object itself, not in parent
			if (matches.hasOwnProperty(key)) {           
				if (matches[key] != null) {
					var num_match = matches[key].length
					if (num_match > max_matches) {
						max_matches = num_match
						workout = key
					} 
				}
			}
		}
	
		if (workout == "schedule") {
			printSchedule();
		} else {
			addMessage("Looks like you want to do " + workout + "! I'll add that"
			+ " to your schedule.", "bot")
			addToSchedule(workout)
			addMessage(responses[Math.floor(Math.random()*responses.length)], "bot")
		}

	}, 1500);
}

function addToSchedule(workout) {
	if (schedule.length == 0) {
		schedule.push(workout)
	} else {
		var insert_index = 0
		var points = workoutMap[workout]
		var added = false
		for (let index = 0; index < schedule.length; index++) {
			const element = schedule[index];
			if (points > workoutMap[element]) {
				schedule.splice(index, 0, workout);
				added = true
				break
			}
		}

		if (!added) {
			schedule[schedule.length] = workout	
		}
	}
}

function printSchedule() {
	var ctime = 0;
	addMessage("Here's your schedule: ", "bot")
	for (let index = 0; index < schedule.length; index++) {
		const element = schedule[index];
		addMessage("Minute " + ctime + ": " + schedule[index], "None")
		ctime += 15
	}
}

$( document ).ready(function() {
	addMessage("Hey there! Let me know what workouts you want to do, and I'll"
		+ " help you schedule your workout! " + emojis[2], "bot")
	addMessage("We intelligently order your workouts so you get the"
		+ " healthiest gainz possible.", "bot")

	var sendButton = document.querySelector('button#sendBtn');
	sendButton.onclick = () => {
		var msg = $("#userText").val()
		$("#userText").val("")
		addMessage(msg, "me") 
		sendButton.disabled = true
		addResponse(msg)
		setTimeout(function(){
			sendButton.disabled = false
		}, 3000);
	}
});

