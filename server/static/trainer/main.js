/**
 * Copyright (C) 2018
 * Brad Huang, Jason Chin, Charlie Lin, and Calvin Woo
 * All rights reserved
 * Special thanks to webRTC project
 */

var video = document.querySelector('video'),
    videoSelect = document.querySelector('select#videoSource');

var selectors = [videoSelect];

var vidCanvas = document.querySelector('#videoCanvas'),
    transmitCanvas = document.querySelector('#transmit'),
    vidCtx = vidCanvas.getContext('2d'),
    transmitCtx = transmitCanvas.getContext('2d'),
    vidCanvasContainer = document.getElementById("videoCanvasContainer"),
    bdy = document.getElementById("bdy");

var startButton = document.querySelector('#start'),
    stopButton = document.querySelector('#stop'),
    resetButton = document.querySelector('#reset');

var minutes = document.querySelector('.minutes'),
    seconds = document.querySelector('.seconds');

var rotate = 0;
var state = 0;
var setCount = 0;
var repCount = 0;
var side = "F"
var deviation = 0;
var critique = ""

var timerTime = 0;
var isRunning = false;
var record = false;
var interval;

var currentWorkout = "none";

const serverURL = "https://localhost:5000/critique";
var img, resultImg;

const CONNECTIONS = [[3, 4, "#ff007f"],
                    [6, 7, "#00ff00"],
                    [2, 3, "#ff7f00"],
                    [5, 6, "#7fff00"],
                    [1, 5, "#ffff00"],
                    [2, 1, "#ff0000"],
                    [12, 13, "#0000ff"],
                    [9, 10, "#007fff"],
                    [11, 12, "00ff7f"],
                    [8, 9, "7f00ff"],
                    [5, 11, "00ffff"],
                    [2, 8, "ff00ff"]];
const SQR_RADIUS = 7;

startButton.onclick = () => {
    console.log(currentWorkout)
    if (currentWorkout === "none") {
        addMessage("You must select a workout to start.")
    } else {
        setTimeout(() => {
            startTimer();
            record = true;
            state = 1;
            upload();
        }, 3000);
    }
}
stopButton.onclick = () => {
    stopTimer();
    record = false;
}
resetButton.onclick = resetTimer;


function adjustSize(){
    vidCtx.canvas.height = Math.min(vidCanvasContainer.offsetHeight, vidCanvasContainer.offsetWidth * 3 / 4);
    vidCtx.canvas.width = vidCtx.canvas.height * 4 / 3;
}


function drawHumanShape(canvas, shapeData){
    var v1, v2, color;

    if (shapeData != -1) {
        
        for (i = 0; i < CONNECTIONS.length; i++){
            v1 = shapeData[CONNECTIONS[i][0]];
            v2 = shapeData[CONNECTIONS[i][1]];
    
            color = CONNECTIONS[i][2];
            drawConnections(canvas.getContext('2d'), v1, v2, canvas.width, canvas.height, color);
        }
    }

}

function drawConnections(ctx, v1, v2, width, height, color){
    x_1 = v1[0] * width;
    y_1 = v1[1] * height;
    x_2 = v2[0] * width;
    y_2 = v2[1] * height;

    ctx.beginPath();
    ctx.moveTo(x_1, y_1);
    ctx.lineTo(x_2, y_2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.stroke();

    ctx.fillRect(x_1 - SQR_RADIUS, y_1 - SQR_RADIUS, 2 * SQR_RADIUS, 2 * SQR_RADIUS);
    ctx.fillRect(x_2 - SQR_RADIUS, y_2 - SQR_RADIUS, 2 * SQR_RADIUS, 2 * SQR_RADIUS);
}


function drawCanvas(video, context, width, height){
    context.drawImage(video, 0, 0, width, height);
    setTimeout(drawCanvas, 50, video, context, width, height);
}


function upload(){
    img = transmitCanvas.toDataURL();

    $.ajax({
        url: serverURL,
        headers: {'Access-Control-Allow-Origin': 'https://localhost:5000' },
        type: 'POST',
        data: {
            image: img,
            workout: currentWorkout,
            state: state,
            side: side,
            rotate: rotate,
            repCount: repCount
        }
    }).done(function(data){
        repCount = data.data.repCount
        critique = data.data.critique
        deviation = data.data.deviation
        state = data.data.state

        console.log(deviation, critique)

        workoutUpdate();

        vidCtx.drawImage(video, 0, 0, vidCanvas.width, vidCanvas.height);
        drawHumanShape(vidCanvas, data["data"]["points"]);
        if (record){
            setTimeout(upload, 10);
        }
    }).fail(function() {
        console.log("Image upload failed!");
        if (record){
            setTimeout(upload, 300);
        }
    })
}


function gotDevices(deviceInfos) {
    // Handles being called several times to update labels. Preserve values.
    var values = selectors.map(function(select) {
        return select.value;
    });

    selectors.forEach(function(select) {
        while (select.firstChild) {
            select.removeChild(select.firstChild);
        }
    });

    for (var i = 0; i !== deviceInfos.length; ++i) {
        var deviceInfo = deviceInfos[i];
        var option = document.createElement('option');
        option.value = deviceInfo.deviceId;
        if (deviceInfo.kind === 'videoinput') {
            option.text = deviceInfo.label || 'Camera ' + (videoSelect.length + 1);
            videoSelect.appendChild(option);
        }
    }

    selectors.forEach(function(select, selectorIndex) {
        if (Array.prototype.slice.call(select.childNodes).some(function(n) {
            return n.value === values[selectorIndex];
        })) {
            select.value = values[selectorIndex];
        }
    });
}


function gotStream(stream) {
    window.stream = stream; // make stream available to console
    video.srcObject = stream;
    // Refresh button list in case labels have become available
    return navigator.mediaDevices.enumerateDevices();
}


function handleError(error) {
    console.log('navigator.getUserMedia error: ', error);
}


function incrementTimer() {
    timerTime++;
  
    const numOfMinutes = Math.floor(timerTime / 60);
    const numOfSeconds = timerTime % 60;
  
    minutes.innerText = pad(numOfMinutes);
    seconds.innerText = pad(numOfSeconds);
}

function startTimer() {
    if (isRunning) return;
    isRunning = true;
    interval = setInterval(incrementTimer, 1000);
}


function stopTimer() {
    if (!isRunning) return;
    isRunning = false;
    clearInterval(interval);
}


function resetTimer() {
    stopTimer();
    $("#chatOut").empty()
    repCount = 0
    timerTime = 0;
    minutes.innerText = '00';
    seconds.innerText = '00';
}


function pad(num) {
    return (num < 10) ? '0' + num : num;
}


function loop() {
    if (!record){
        vidCtx.drawImage(video, 0, 0, vidCanvas.width, vidCanvas.height);
    }
    setTimeout(loop, 50);
}


function start() {
    videoSource = videoSelect.value;
    var constraints = {
      audio: false,
      video: {deviceId: videoSource ? {exact: videoSource} : undefined}
    };

    navigator.getMedia = navigator.mediaDevices.getUserMedia(constraints) ||
                         navigator.mediaDevices.webkitGetUserMedia(constraints) ||
                         navigator.mediaDevices.mozGetUserMedia(constraints) ||
                         navigator.mediaDevices.msGetUserMedia(constraints);

    navigator.getMedia.
        then(gotStream).then(gotDevices).catch(handleError);

    adjustSize();
    loop();
    drawCanvas(video, transmitCtx, transmitCanvas.width, transmitCanvas.height);
}


videoSelect.onchange = start;
start();

/*

Workout Management

*/
function changeWorkout() {
    resetTimer();
    record=false;
    currentWorkout = $("#workoutSelector").val()
    $("#chatOut").empty()
    switch (currentWorkout) {
        case "pushup":
            addMessage("Please set up your pushups with your right shoulder facing the camera.")
            rotate = 90;
            side = "R"
            repCount = 0
            break;

        case "plank":
            addMessage("Please set up your planks with your right shoulder facing the camera.")
            rotate = 90;
            side = "R"
            repCount = 0
            break;

        case "curls":
            addMessage("Please set up your curls with your right shoulder facing the camera."
            + " After 10 repetitions, we'll switch to the left hand."
            + " Only curl with the arm closest to the camera.")
            rotate = 0;
            side = "R"
            repCount = 0
            break;

        case "shoulderpress":
            addMessage("Please set up your shoulder presses facing the camera.")
            rotate = 0;
            side = "F"
            repCount = 0
            break;

        case "squats":
            addMessage("Please perform squats perfectly perpendicular to the camera.")
            rotate = 0;
            side = "R"
            repCount = 0
            break;

        default:
            break;
    }
    workoutUpdate()
}

function rgb(r, g, b) {
    if (r < 0) {
        r = 0
    } else if (r > 255) {
        r = 255
    }

    if (g < 0) {
        g = 0
    } else if (g > 255) {
        g = 255
    }

    if (b < 0) {
        b = 0
    } else if (b > 255) {
        b = 255
    }
    return "rgb(" + Math.floor(r) + "," + Math.floor(g) + "," + "0)"
}

function workoutUpdate() {
    $("#repCounter").text(repCount)
    $("#deviationDisplay").text(deviation)
    if (state == 1) {
        $("#stateDisplay").text("Go down")
    } else if (state == 2) {
        $("#stateDisplay").text("Go up")    
    } else {
        $("#stateDisplay").text("Rest")
    } 
    
    if (side == "R") {
        $("#sideDisplay").text("Right")
    } else if (side == "L") {
        $("#sideDisplay").text("Left")
    } else if (side == "F") {
        $("#sideDisplay").text("Center")
    }


    if (record) {
        addMessage(critique)
    }

    var clr = rgb(255*deviation, (255 - 255*deviation), 0);
    document.getElementById("indicator").style.backgroundColor = clr;

    switch (currentWorkout) {
        case "pushup":
        case "plank":
            if (repCount >= 10) {
                addMessage("Nice job! Take a quick break and click start when you're ready for the next set!")
                rotate = 90;
                side = "R"
                repCount = 0
                record = false
            }
            break;

        case "curls":
            if (repCount >= 10) {
                addMessage("Nice job! Take a quick break and click start when you're ready for the next set!")
                rotate = 0;
                repCount = 0
                record = false
                if (side == "R") {
                    side = "L"
                } else {
                    side = "R"
                }
            }
            break;

        case "squats":
            if (repCount >= 10) {
                addMessage("Nice job! Take a quick break and click start when you're ready for the next set!")
                rotate = 0;
                side = "R"
                repCount = 0
                record = false
            }
           break;

        case "shoulderpress":
            if (repCount >= 10) {
                addMessage("Nice job! Take a quick break and click start when you're ready for the next set!")
                rotate = 0;
                side = "F"
                repCount = 0
                record = false
            }
            break;

        default:
            break;
    }
}

/*

Message Box

*/

function addMessage(msg, style) {
    var msgBox = $("#chatOut")
    msgBox.text(msg);
    updateScroll();
}

function updateScroll(){
    var element = document.getElementById("chatOut");
    element.scrollTop = element.scrollHeight;
}