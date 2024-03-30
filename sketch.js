// based on The Coding Train Coding Challenge #104
// https://www.youtube.com/watch?v=dLp10CFIvxI
//
// 16:00 - 35:30
// write optimize function
// write loss function

let x_vals = [];
let y_vals = [];
let m;
let b;

const learningRate = 0.3;
const optimizer = tf.train.sgd(learningRate);
let firstError = 0;
let currentError = 0;

function setup() {
  const canvas = createCanvas(400,400);
  canvas.parent("canvas")
  m = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(1)));
}

function draw() {
  background(64);  
  drawVals();
  drawPrediction();
  if(x_vals.length > 0) {

    optimizer.minimize(loss);
    
    // display initial loss
    noStroke();
    fill(200);
    textSize(20);
    text(firstError.toFixed(8), width-10, 25);  

    // display optimized loss
    noStroke();
    fill(200, 100, 100);
    textSize(20);
    textAlign(RIGHT);
    text(currentError.toFixed(8), width-10, 50);
  }
    /* console.log(frameCount + ": " + tf.memory().numTensors); */
}

function mousePressed() {
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
  
    const x = map(mouseX, 0, width, 0, 1);
    const y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y);
    firstError = 0; 
}}

function loss() {
  const predictions = tf.tidy(predict);
  const actuals = tf.tensor1d(y_vals);  
  const error = predictions.sub(actuals).square().mean();
  currentError = error.dataSync()[0];
  if (firstError == 0) {
    firstError = currentError;
  }
  return error;
}

function predict() {
  const xs = tf.tensor1d(x_vals);  
  const ys = xs.mul(m).add(b);
  const ys_temp = ys.dataSync();

  for (let i = 0; i < x_vals.length; i++) {
    // draw given x and predicted y
    stroke(128, 128, 0);
    strokeWeight(8);
    const x = map(x_vals[i], 0, 1, 0, width);
    const y_pred = map(ys_temp[i], 0, 1, height, 0);
    point(x, y_pred);
    
    // and draw a line to given x and given y
    stroke(200, 100, 100);
    strokeWeight(2);
    const y_given = map(y_vals[i], 0, 1, height, 0);
    line(x, y_pred, x, y_given);
  }
  
  return ys;
}

function drawVals() {
  stroke(128);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    const x = map(x_vals[i], 0, 1, 0, width);
    const y = map(y_vals[i], 0, 1, height, 0);
    point(x, y);
  }
}

function drawPrediction() {
  stroke(128, 128, 0);
  strokeWeight(2);
  const y1 = map(b.dataSync()[0], 0, 1, height, 0);
  const y2 = map(m.dataSync()[0], 0, 1, 0, -width) + y1;
  line(0, y1, 400, y2);
}