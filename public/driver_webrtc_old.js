// WebRTC
//import '@tensorflow/tfjs';
//import * as cocoSsd from '@tensorflow-models/coco-ssd';


export function DriverWebRTC(iceConfig, log, sendToServer, hangUpCall) {

  var pc = null;
  var localVideo = document.getElementById("localVideo");
  var remoteVideo = document.getElementById("remoteVideo");
  var remoteView = document.getElementById("remoteView");
  var localView = document.getElementById("localView");
  var children = [];
  var canvas = document.getElementById("canvas");
  var vidWidth = canvas.width;
  var vidHeight = canvas.height;
  var ctx = canvas.getContext("2d");
  var posx = 0;
  var posy = 0;
  var targetA = 0;
  var turn = false;
  var move = false;
  
  var class_names=[
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
  ];
  
  //ctx.fillStyle = "blue";
  //ctx.fill();
  ctx.strokeRect(0,0,canvas.width,canvas.height);
  
  
  /*cocoSsd.load().then(function (loadedModel) {
      model = loadedModel;
  });*/
  var model = undefined;
  const model_url = 'https://raw.githubusercontent.com/KostaMalsev/ImageRecognition/master/model/mobile_netv2/web_model2/model.json';
  //Call load function
  asyncLoadModel(model_url);
  
  async function asyncLoadModel(model_url) {
    model = await tf.loadGraphModel(model_url);
    console.log('Model loaded');
  }

  function predictWebcamTF() {
    turn = false;
    
    detectTFMOBILE(remoteVideo).then(function () {
      window.requestAnimationFrame(predictWebcamTF);
    });
  }
    
  const imageSize = 480; 
  var classProbThreshold = 0.66;
  async function detectTFMOBILE(img) {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(remoteVideo, 0, 0, canvas.width, canvas.height);
    console.log("H: " + vidHeight + ", W: " + vidWidth);
    await tf.nextFrame();
    const tfImg = tf.browser.fromPixels(img);
    const preprocessedInput = tfImg.expandDims();
    
    //const prediction = model.predict(preprocessedInput);
    const smallImg = tf.image.resizeBilinear(tfImg, [vidHeight,vidWidth]);
    const resized = tf.cast(smallImg, 'int32');
    var tf4d_ = tf.tensor4d(Array.from(resized.dataSync()), [1,vidHeight, vidWidth, 3]);
    const tf4d = tf.cast(tf4d_, 'int32');
    //Perform the detection with your layer model:
    let predictions = await model.executeAsync(preprocessedInput);
    //Draw box around the detected object:
    renderPredictionBoxes(predictions[4].dataSync(), predictions[1].dataSync(), predictions[2].dataSync(), predictions);
    //Dispose of the tensors (so it won't consume memory)
    tfImg.dispose();
    smallImg.dispose();
    resized.dispose();
    tf4d.dispose();
  }
    
    
  function renderPredictionBoxes (predictionBoxes, predictionClasses, predictionScores, predictions) {
      for (let i = 0; i < children.length; i++) {
        remoteView.removeChild(children[i]);
      }
      children.splice(0);

      // Now lets loop through predictions and draw them to the live view if
      // they have a high confidence score.
      for (let i = 0; i < predictions.length; i++) {
        console.log(i)
        const minY = (predictionBoxes[i * 4] * vidHeight).toFixed(0);
        const minX = (predictionBoxes[i * 4 + 1] * vidWidth).toFixed(0);
        const maxY = (predictionBoxes[i * 4 + 2] * vidHeight).toFixed(0);
        const maxX = (predictionBoxes[i * 4 + 3] * vidWidth).toFixed(0);
        const score = predictionScores[i * 3] * 100;
        const width_ = (maxX-minX).toFixed(0);
        const height_ = (maxY-minY).toFixed(0);
        
        // If we are over 66% sure we are sure we classified it right, draw it!
        if (score > 66) {
          console.log(predictions[2].dataSync());
          const predClass = class_names[i]; // you can also use arraySync or their equivalents async methods
          console.log('Predictions: ', predClass);
          
          const p = document.createElement('p');
          
          p.innerText = predClass  + ' - with ' 
              + Math.round(score) + '% confidence.';
          
          /*p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
              + (predictions[n].bbox[1] - 5) + 'px; width: ' 
              + (predictions[n].bbox[2] - 5) + 'px; top: 0; left: 0;';

          const highlighter = document.createElement('div');
          highlighter.setAttribute('class', 'highlighter');
          highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
              + predictions[n].bbox[1] + 'px; width: ' 
              + predictions[n].bbox[2] + 'px; height: '
              + predictions[n].bbox[3] + 'px;';*/

          //remoteView.appendChild(highlighter);
          remoteView.appendChild(p);
          //children.push(highlighter);
          children.push(p);
          //ctx.fillStyle = "#FF0000";
          
          ctx.beginPath();
          ctx.font = "15px Arial";
          ctx.fillStyle = "red";
          ctx.strokeStyle = "red";
          //ctx.fillText(predClass +": " + score + "%", minX, minY);
          ctx.strokeRect(0,0,10,10);
          //ctx.strokeRect(minX, minY, width_, height_);
          //ctx.fillRect(posx,posy,3,3);*/
          console.log("x: " + minX + ", y: " + minY + ", w: " + width_ + ", h: " + height_);
        }
      }

      // Call this function again to keep predicting when the browser is ready.
      /*setTimeout(function(){
          console.log("pos:" + posx + ", " + posy + "\n area: " + targetA + "\n turn: " + turn + "\n move: " + move);
      }, 2000);
      window.requestAnimationFrame(predictWebcam);
    });*/
  }
  
  
  this.handleVideoOffer = async (msg) => {
    log("Received call offer");

    pc = new RTCPeerConnection(iceConfig);
    pc.onicecandidate = (event) => this.onicecandidate(event);
    pc.oniceconnectionstatechange = () => this.oniceconnectionstatechange();
    pc.onicegatheringstatechange = () => this.onicegatheringstatechange();
    pc.onsignalingstatechange = () => this.onsignalingstatechange();
    pc.ontrack = (event) => this.ontrack(event);

    var desc = new RTCSessionDescription(msg);
    await pc.setRemoteDescription(desc);

    if (!localVideo.srcObject) {
      localVideo.srcObject = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
    }

    localVideo.srcObject.getTracks().forEach(track => pc.addTrack(track, localVideo.srcObject));

    await pc.setLocalDescription(await pc.createAnswer());
    sendToServer(pc.localDescription);

    log("Sending SDP answer");
  }

  this.handleCandidate = (candidate) => {
    var candidate = new RTCIceCandidate(candidate);
    log("Adding received ICE candidate: " + JSON.stringify(candidate));
    pc.addIceCandidate(candidate);
  }

  this.closeVideoCall = () => {
    log("Closing the call");

    if (pc) {
      log("Closing the peer connection");

      pc.onicecandidate = null;
      pc.oniceconnectionstatechange = null;
      pc.onicegatheringstatechange = null;
      pc.onsignalingstatechange = null;
      pc.ontrack = null;

      pc.getSenders().forEach(track => { pc.removeTrack(track); });
      
      if (remoteVideo) {
        remoteVideo.srcObject = null;
        remoteVideo.controls = false;
      }
      
      pc.close();
      pc = null;
    }
  }

  this.onicecandidate = (event) => {
    if (event.candidate) {
      log("Outgoing ICE candidate: " + event.candidate.candidate);
      sendToServer({
        type: "candidate",
        sdpMLineIndex: event.candidate.sdpMLineIndex,
        sdpMid: event.candidate.sdpMid,
        candidate: event.candidate.candidate
      });
    }
  };

  this.oniceconnectionstatechange = () => {
    log("ICE connection state changed to " + pc.iceConnectionState);
    switch(pc.iceConnectionState) {
      case "closed":
      case "failed":
      case "disconnected":
        hangUpCall();
        break;
    }
  };

  this.onicegatheringstatechange = () => {
    log("ICE gathering state changed to " + pc.iceGatheringState);
  };

  this.onsignalingstatechange = () => {
    log("WebRTC signaling state changed to: " + pc.signalingState);
    switch(pc.signalingState) {
      case "closed":
        hangUpCall();
        break;
    }
  };

  this.ontrack = (event) => {
    log("Track event");
    remoteVideo.srcObject = event.streams[0];
    remoteVideo.controls = true;
    /*cocoSsd.load().then(function (loadedModel) {
      model = loadedModel;
    });*/
    remoteVideo.addEventListener('loadeddata', predictWebcamTF);
    //console.log("pos:" + posx + posy);
  };  
};

export default DriverWebRTC;
