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
	<!--<script src="./onnx.min.js"></script>-->
	<script src="./jquery-3.6.0.min.js"></script>
	<link rel="stylesheet" href="main.css" type="text/css" />
</head>
<body>

<style>
.app{
	width: 500px;
	margin-left: auto;
	margin-right: auto;
}
canvas{
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
.app__result_prob_title{
	padding-top: 10px;
	font-size: 18px;
}
.app__result_prob_row{
	padding-top: 5px;
	padding-bottom: 5px;
	font-size: 18px;
}
.app__result_prob_row.select{
	background-color: #0ef10e;
}
.button{
	padding: 6px 12px;
	margin-bottom: 10px;
}
#canvas2{
	/*display: none;*/
}
</style>

<div class="app">
	
	<h1>Проверка модели</h1>
	
	<div class="app__info">Нарисуйте цифру мышкой. Левая кнопка рисует, правая стирает</div>
	
	<button type="button" class="button button--clear">Clear</button>
	
	<div class="app__canvas_wrap">
		<div class="app__canvas">	
			<canvas id='canvas' width="256" height="256"></canvas>
			<div>Конвертация:</div>
			<canvas id='canvas2' width="28" height="28"></canvas>
		</div>
		<div class="app__result">
			<div class="app__result_title">Ответ: <span class="app__result_value"></span></div>
			<div class="app__result_prob_title">Вероятность:</div>
			<div class="app__result_prob">
				<div class="app__result_prob_row" data-index="1">1 - <span></span></div>
				<div class="app__result_prob_row" data-index="2">2 - <span></span></div>
				<div class="app__result_prob_row" data-index="3">3 - <span></span></div>
				<div class="app__result_prob_row" data-index="4">4 - <span></span></div>
				<div class="app__result_prob_row" data-index="5">5 - <span></span></div>
				<div class="app__result_prob_row" data-index="6">6 - <span></span></div>
				<div class="app__result_prob_row" data-index="7">7 - <span></span></div>
				<div class="app__result_prob_row" data-index="8">8 - <span></span></div>
				<div class="app__result_prob_row" data-index="9">9 - <span></span></div>
				<div class="app__result_prob_row" data-index="0">0 - <span></span></div>
			</div>
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
canvas.addEventListener("contextmenu", onMouse("contextmenu") );

$('.button--clear').click(function(){
	canvas_ctx.clearRect(0, 0, canvas.width, canvas.height);
	canvas2_ctx.clearRect(0, 0, canvas2.width, canvas2.height);
	$('.app__result_value').html('');
	$('.app__result_prob_row').removeClass('select');
	$('.app__result_prob_row span').html('');
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
	let file = './mnist3.onnx';
	
	file = './4new.onnx';
	file = './mnist4.onnx';
	
	model = await ort.InferenceSession.create(file, {
		//"executionProviders": ["webgl"]
	});
}

async function predict1(input)
{
	input = Float32Array.from(input);
	input = new ort.Tensor('float32', input, input_shape);
	let res = await model.run({ 'input': input });
	let output = res['output'].data;
	return output;
}

function getRGBAColor(data, pos)
{
	let color = Math.round((data[pos*4 + 0] + data[pos*4 + 1] + data[pos*4 + 2]) / 3);
	if (data[pos*4 + 3] > 50)
	{
		if (color > 50) return 0;
		else return 1;
	}
	
	return 0;
}

function getImageBox()
{
	let left = 0;
	let right = 255;
	let top = 0;
	let bottom = 255;
	
	function isEmptyRow(data, y)
	{
		for (let x=0; x<256; x++)
		{
			let color = getRGBAColor(data, y*256 + x);
			if (color == 1)
			{
				return false;
			}
		}
		return true;
	}
	
	function isEmptyCol(data, x)
	{
		for (let y=0; y<256; y++)
		{
			let color = getRGBAColor(data, y*256 + x);
			if (color == 1)
			{
				return false;
			}
		}
		return true;
	}
	
	let data = canvas_ctx.getImageData(0, 0, 256, 256).data;
	
	/* Определяем границы цифры */
	while ( left < 256 && isEmptyCol(data, left) ) left++;
	while ( right >= 0 && isEmptyCol(data, right) ) right--;
	while ( top < 256 && isEmptyRow(data, top) ) top++;
	while ( bottom >= 0 && isEmptyRow(data, bottom) ) bottom--;
	
	console.log( left, top, right, bottom );
	
	let x = left;
	let y = top;
	let w = right - left + 1;
	let h = bottom - top + 1;
	let res = { x: x, y: y, w: w, h: h };
	let max = 0;
	
	if (w > h)
	{
		res["y"] = y - Math.round((w - h) / 2);
		res["h"] = w;
		max = w;
	}
	else
	{
		res["x"] = x - Math.round((h - w) / 2);
		res["w"] = h;
		max = h;
	}
	
	max = Math.round(max * 0.1);
	
	res["x"] -= max;
	res["y"] -= max;
	res["h"] += max*2;
	res["w"] += max*2;
	console.log( res );
	
	return res;
}

function getImage()
{
	let box = getImageBox();
	
	canvas2_ctx.clearRect(0, 0, canvas2.width, canvas2.height);
	canvas2_ctx.drawImage(canvas, box.x, box.y, box.w, box.h, 0, 0, 28, 28);
	
	let data = canvas2_ctx.getImageData(0, 0, 28, 28).data;
	let input = [];
	
	for (let i=0; i<28*28; i++)
	{
		let color = getRGBAColor(data, i);
		input.push(color);
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
	
	output = output.sort(function(a, b){
		if (a.value > b.value) return -1;
		if (a.value < b.value) return 1;
		return 0
	});
	
	let sum = output.reduce(
		(s, item) => {
			return item.value > 0 ? s + item.value : s
		}, 0
	);
	
	let result_value = output[0].index;
	$('.app__result_value').html(result_value);
	
	let app_row_class = [];
	for (index in output)
	{
		let item = output[index];
		if (item.value > 0)
		{
			$('.app__result_prob_row[data-index=' + item.index + '] span').html(
				Math.round( item.value / sum * 100 ) + '%'
			);
		}
	}
	$('.app__result_prob_row').removeClass('select');
	$('.app__result_prob_row[data-index=' + result_value + ']').addClass('select');
}

init1();

</script>
	
</body>
</html>