<?php

$uri = isset($_SERVER["REQUEST_URI"]) ? $_SERVER["REQUEST_URI"] : "";
$uri_arr = parse_url($uri);
$uri_path = isset($uri_arr["path"]) ? $uri_arr["path"] : "";

$uri_path = preg_replace("/^\/+/", "", $uri_path);
$uri_path = preg_replace("/\.\./", "", $uri_path);

/* Раздача статических файлов */
if (file_exists($uri_path))
{
	$uri_path_info = pathinfo($uri_path);
	
	$extension = isset($uri_path_info['extension']) ? $uri_path_info['extension'] : '';
	if ($extension == "wasm")
	{
		header("Content-type: application/wasm");
	}
	$content=file_get_contents($uri_path);
	echo $content;
	exit();
}

?><!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8" />
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<meta http-equiv="X-UA-Compatible" content="IE=edge" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Проверка модели</title>
	<script src="./ort.min.js"></script>
	<script src="./onnx.min.js"></script>
	<script src="./jquery-3.6.0.min.js"></script>
	<link rel="stylesheet" href="main.css" type="text/css" />
</head>
<body>

<style>
.content{
	width: 500px;
	margin-left: auto;
	margin-right: auto;
}
.app__canvas{
	border: 2px solid black;
}
.app__info{
	padding-bottom: 50px;
}
.app__canvas_wrap{
	display: flex;
}
.app__result{
	padding-left: 50px;
}
.app__result_title{
	padding-bottom: 20px;
	font-size: 20px;
}
.button{
	padding: 6px 12px;
	margin-bottom: 10px;
}
</style>

<div class="content app">
	
	<h1>Проверка модели</h1>
	
	<div class="app__info">Нарисуйте цифру мышкой. Левая кнопка рисует, правая стирает</div>
	
	<button type="button" class="button button--clear">Clear</button>
	
	<div class="app__canvas_wrap">
		<canvas id='canvas' class="app__canvas" width="128" height="128"></canvas>
		<div class="app__result">
			<canvas id='canvas2' class="app__canvas" width="28" height="28"></canvas>
			<div class="app__result_title">Ответ: <span class="app__result_value"></span></div>
		</div>
	</div>
	
</div>
	
<script>
let is_draw = false;
let model = null;
let canvas = document.getElementById('canvas');
let canvas_ctx = canvas.getContext("2d");
let canvas2_ctx = canvas2.getContext("2d");
let prev_x = 0;
let prev_y = 0;
let line_width = 16;
let input_shape = [1,28,28];

canvas.addEventListener("mousemove", onMouse("mousemove") );
canvas.addEventListener("mousedown", onMouse("mousedown") );
document.addEventListener("mouseup", onMouse("mouseup") );
document.addEventListener("contextmenu", onMouse("contextmenu") );

$('.button--clear').click(function(){
	canvas_ctx.clearRect(0, 0, canvas.width, canvas.height);
	canvas2_ctx.clearRect(0, 0, canvas2.width, canvas2.height);
	$('.app__result_value').html('');
});

function drawLine(x1, y1, x2, y2, color)
{
	canvas_ctx.beginPath();
	canvas_ctx.fillStyle = color;
	canvas_ctx.strokeStyle = color;
	if (color == "white") canvas_ctx.lineWidth = line_width + 5;
	else canvas_ctx.lineWidth = line_width;
	canvas_ctx.lineCap = 'round';
	canvas_ctx.moveTo(x1, y1);
    canvas_ctx.lineTo(x2, y2);
	canvas_ctx.stroke();
	canvas_ctx.closePath();
}

function onMouse(event_name)
{
	return function (e)
	{
		let x = e.offsetX;
		let y = e.offsetY;
		let color = "black";
		
		if (e.which == 3)
		{
			color = "white";
		}
		if (event_name == "mouseup" || event_name == "mouseout")
		{
			if (is_draw) recognizeImage();
			is_draw = false;
		}
		if (event_name == "mousedown") is_draw = true;
		
		if (is_draw && event_name == "mousedown")
		{
			prev_x = x;
			prev_y = y;
		}
		else if (is_draw && event_name == "mousemove")
		{
			drawLine(prev_x, prev_y, x, y, color);
			prev_x = x;
			prev_y = y;
		}
		if (event_name == "contextmenu")
		{
			e.preventDefault();
		}
	}
}

async function init1()
{
	model = await ort.InferenceSession.create('./mnist2.onxx', {
		"executionProviders": ["webgl"]
	});
}

async function init2()
{
	model = new onnx.InferenceSession({ backendHint: "webgl" });
	await model.loadModel("./mnist2.onxx");
}

async function predict1(input)
{
	input = Float32Array.from(input);
	input = new ort.Tensor('float32', input, input_shape);
	let res = await model.run({ 'input': input });
	let output = res['output'].data;
	return output;
}

async function predict2(input)
{
	//input = Float32Array.from(input);
	let output = await model.run([ input ]);
	let tensor = output.values().next().value;
	let data = tensor.data;
	return data;
}

function getImage()
{
	canvas2_ctx.drawImage(canvas, 0, 0, 28, 28);
	
	let data = canvas2_ctx.getImageData(0, 0, 28, 28).data;
	let input = [];
	/*
	for (let y=0; y<28; y++)
	{
		let row = [];
		for (let x=0; x<28; x++)
		{
			row.push(0);
		}
		//row = Float32Array.from(row);
		input.push( row );
	}
	
	for (let y=0; y<28; y++)
	{
		for (let x=0; x<28; x++)
		{
			let i = y*28 + x;
			let color = Math.round((data[i*4 + 0] + data[i*4 + 1] + data[i*4 + 2]) / 3) / 256;
			if (data[i*4 + 3] > 50)
			{
				if (color < 50) input[y][x] = 1;
			}
		}
	}
	*/
	
	for (let i=0; i<28*28; i++)
	{
		let color = Math.round((data[i*4 + 0] + data[i*4 + 1] + data[i*4 + 2]) / 3) / 256;
		if (data[i*4 + 3] > 50)
		{
			if (color > 50) input.push(0);
			else input.push(1);
		}
		else
		{
			input.push(0);
		}
	}
	return input;
}

async function recognizeImage()
{
	let res = null;
	let input = getImage();
	
	let output = await predict1(input);
	
	output = Array.from(output);
	output = output.map(function(value, index){
		return {
			"index": index,
			"value": value,
		}
	});
	
	//console.log(output);
	
	output = output.sort(function(a, b){
		if (a.value > b.value) return -1;
		if (a.value < b.value) return 1;
		return 0
	});
	
	$('.app__result_value').html(output[0].index);
	
	//console.log(output);
	
}

init1();

</script>
	
</body>
</html>