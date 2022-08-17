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
.app__table{
	padding-left: 50px;
}
.app__table_label, .app__table_value{
	padding: 10px;
}
.app__table_label{
	font-size: 20px;
}
.app__table_value{
	width: 100px;
}
</style>

<div class="content app">
	
	<h1>Проверка модели</h1>
	
	<div class="app__info">Нарисуйте цифру мышкой. Левая кнопка рисует, правая стирает</div>
	
	<div class="app__canvas_wrap">
		<canvas id='canvas' class="app__canvas" width="256" height="256"></canvas>
		<table class="app__table">
			<?php for ($i=0; $i<5; $i++){ ?>
			<tr>
				<td class="app__table_label"><?= $i ?></td>
				<td class="app__table_value" data-value="<?= $i ?>"></td>
				<td class="app__table_label"><?= $i + 5 ?></td>
				<td class="app__table_value" data-value="<?= $i + 5 ?>"></td>
			</tr>
			<?php } ?>
		</table>
	</div>
	
</div>
	
<script>
let is_draw = false;
let model = null;
let canvas = document.getElementById('canvas');
let canvas_ctx = canvas.getContext("2d");
let prev_x = 0;
let prev_y = 0;
let line_width = 10;

canvas.addEventListener("mousemove", onMouse("mousemove") );
canvas.addEventListener("mousedown", onMouse("mousedown") );
document.addEventListener("mouseup", onMouse("mouseup") );
document.addEventListener("contextmenu", onMouse("contextmenu") );

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
		if (event_name == "mouseup" || event_name == "mouseout") is_draw = false;
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


async function init_onxx()
{
	model = await ort.InferenceSession.create('./mnist.onxx');
}

init_onxx();

</script>
	
</body>
</html>