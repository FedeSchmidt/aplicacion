import { Component, OnInit } from '@angular/core';
import { Auxiliar } from './auxiliar';
import * as tf from '@tensorflow/tfjs';
import { Layer } from './../models/layer';
import Chart from 'chart.js';
import { JsonPipe } from '@angular/common';

@Component({
	selector: 'app-mnist',
	templateUrl: './mnist.component.html',
	styleUrls: [ './mnist.component.scss' ]
})
export class MnistComponent implements OnInit {
	// Variables para el dataset
	MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
	MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
	IMAGE_SIZE = 784;
	NUM_CLASSES = 10;
	NUM_DATASET_ELEMENTS = 65000;
	NUM_TRAIN_ELEMENTS = 55000;
	NUM_TEST_ELEMENTS = this.NUM_DATASET_ELEMENTS - this.NUM_TRAIN_ELEMENTS;
	IMAGE_H = 28;
	IMAGE_W = 28;

	//codigo equivalente
	model_code = [];
	train_code = [];
	compile_code = [];

	//ejemplos
	sliceEjemplos = 0;
	modelo_entrenado = false;
	predictions: any;
	data;

	//Entrenamiento
	entrenando = false;

	datasetImages: Float32Array;
	datasetLabels: Uint8Array;
	trainImages: any;
	testImages: any;
	trainLabels: any;
	testLabels: any;

	net = [];
	max_capas = 8;
	cant_capas = 0;
	model: any;

	progreso = 0;
	validation_accuracy = '---';
	test_accuracy = '---';
	trainset_accuracy = '---';

	arr = [];
	prediction: any;

	cantLayers = 1;

	painting = false;
	canvas;
	cx;

	// parámetros generales
	learning_ratio = 0.15;
	epochs = 10;
	epochActual = 0;
	//metrics = 'accuracy';
	batch_check = true;
	batch_size = 120;

	resultados = false;

	metricas = [ 'accuracy', 'mean_squared_error' ];
	metric = 'accuracy';
	loss_functions = [ 'mean_squared_error', 'categorical_crossentropy' ];
	cost_function = 'categorical_crossentropy';

	regularization = 'Ninguno';
	regularization_ratio = 0.1;

	//Nodos
	tipos_capas = [ 'Dense', 'Dropout' ];
	activations = [ 'ReLU', 'Sigmoid', 'Linear', 'Softmax' ];

	examples_test = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ];
	ejemplos_cargados = false;
	prediction_labels;
	prediction_nuevas;
	testExamplesLength = 0;
	test_data;

	img_src = '';
	dataImg;

	mostrar_codigo = true;
	mostrar_ejemplos = false;
	mostrar_canvas = false;
	canvases: any = [];

	res_obtenidos = {};
	hayResultados = false;
	imagenes_cargadas = false;

	//graficos de entrenamiento
	dataLossChart = {
		labels: [],
		datasets: [
			{
				label: 'Loss',
				data: [],
				borderColor: 'rgba(209, 10, 46, 1)',
				pointBackgroundColor: 'rgba(209, 10, 46, 1)',
				pointBorderColor: 'rgba(209, 10, 46, 1)'
			},
			{
				label: 'Val-Loss',
				data: [],
				borderColor: 'rgba(4, 158, 53, 1)',
				pointBackgroundColor: 'rgba(4, 158, 53, 1)',
				pointBorderColor: 'rgba(4, 158, 53, 1)'
			}
		]
	};

	dataAccChart = {
		labels: [],
		datasets: [
			{
				label: 'Accuracy',
				data: [],
				borderColor: 'rgba(209, 10, 46, 1)',
				pointBackgroundColor: 'rgba(209, 10, 46, 1)',
				pointBorderColor: 'rgba(209, 10, 46, 1)'
			},
			{
				label: 'Val-Accuracy',
				data: [],
				borderColor: 'rgba(4, 158, 53, 1)',
				pointBackgroundColor: 'rgba(4, 158, 53, 1)',
				pointBorderColor: 'rgba(4, 158, 53, 1)'
			}
		]
	};

	optionsLossChart = {
		title: {
			display: true,
			text: 'Loss',
			fontSize: 16,
			fontColor: 'rgba(0,0,0,1)'
		},
		scales: {
			xAxes: [
				{
					ticks: {
						display: false
					},
					gridLines: {
						display: false
					}
				}
			]
		}
	};

	optionsAccChart = {
		title: {
			display: true,
			text: 'Accuracy',
			fontSize: 16,
			fontColor: 'rgba(0,0,0,1)'
		},
		scales: {
			xAxes: [
				{
					ticks: {
						display: false
					},
					gridLines: {
						display: false
					}
				}
			]
		}
	};
	accuracyChart: any;
	lossChart: any;

	constructor() {}

	ngOnInit() {
		this.armarRedBase();

		var canv = <HTMLCanvasElement>document.getElementById('loss-chart');
		var ctx = canv.getContext('2d');

		this.lossChart = new Chart(ctx, {
			type: 'line',
			data: this.dataLossChart,
			options: this.optionsLossChart
		});

		var canv = <HTMLCanvasElement>document.getElementById('accuracy-chart');
		var ctx2 = canv.getContext('2d');

		this.accuracyChart = new Chart(ctx2, {
			type: 'line',
			data: this.dataAccChart,
			options: this.optionsAccChart
		});

		this.load().then(() => {
			console.log('Imágenes Cargadas');
			this.cargarEjemplos(0);
		});
		//const auxiliar = new Auxiliar();
		// var canv = <HTMLCanvasElement>document.getElementById('test-chart');
		// var ctx = canv.getContext('2d');
	}

	armarRedBase() {
		this.net.push(new Layer('Flatten', 784, '-', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0));
		this.net.push(new Layer('Dense', 42, 'ReLU', null, 0));
		this.net.push(new Layer('Dense', 10, 'Softmax', null, 0));
		this.cant_capas = this.net.length;

		this.actualizarCodigo1();
		this.actualizarCodigo();
	}

	showCodigos() {
		this.mostrar_codigo = !this.mostrar_codigo;
	}

	cargarEjemplos(diff) {
		if (diff < 0) {
			this.sliceEjemplos = this.sliceEjemplos - 20;
			console.log(this.sliceEjemplos);
		}
		console.log('Cargando nuevos ejemplos ' + this.sliceEjemplos);

		this.imagenes_cargadas = false;
		this.data = this.getTestData(10);

		const labels = Array.from(this.data.labels.argMax(1).dataSync());
		this.prediction_labels = labels;
		this.imagenes_cargadas = true;
		const testExamples = this.data.xs.shape[0];

		for (let i = 0; i < testExamples; i++) {
			const image = this.data.xs.slice([ i, 0 ], [ 1, this.data.xs.shape[1] ]);
			let div = document.getElementById('div_canvases_' + i);
			let canvas = document.getElementById('canvast' + i);
			this.draw(image.flatten(), canvas);
			canvas.classList.add('prediction-canvas');
			if (document.getElementById('real' + i) == undefined) {
				let real = document.createElement('h6');
				real.id = 'real' + i;
				real.innerHTML = 'Real: ' + this.prediction_labels[i];
				div.appendChild(real);
			} else {
				document.getElementById('real' + i).innerHTML = 'Real: ' + this.prediction_labels[i];
			}
			if (this.modelo_entrenado) {
				let y_pred = this.model.predict(this.data.xs);
				this.predictions = Array.from(y_pred.argMax(1).dataSync());
				if (document.getElementById('prediccion' + i) == undefined) {
					let prediccion = document.createElement('h6');
					prediccion.id = 'prediccion' + i;
					prediccion.innerHTML = 'Predicción: ' + this.predictions[i];
					div.appendChild(prediccion);
				} else {
					document.getElementById('prediccion' + i).innerHTML = 'Predicción: ' + this.predictions[i];
				}

				if (this.prediction_labels[i] !== this.predictions[i]) {
					canvas.classList.remove('pred-bien');
					canvas.classList.add('pred-mal');
				} else {
					canvas.classList.remove('pred-mal');
					canvas.classList.add('pred-bien');
				}
			}
		}
	}
	showEjemplos() {
		this.mostrar_ejemplos = !this.mostrar_ejemplos;
	}

	showCanvas() {
		this.mostrar_canvas = !this.mostrar_canvas;
		// this.canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
		// this.cx = this.canvas.getContext('2d');

		// this.canvas.addEventListener('mousedown', this.startPosition);
		// this.canvas.addEventListener('mouseup', this.finishedPosition);
		// this.canvas.addEventListener('mousemove', this.mouseMove);
	}

	async guardarModelo() {
		let obj = {
			learning_ratio: this.learning_ratio,
			epochs: this.epochs,
			batch_check: this.batch_check,
			batch_size: this.batch_size,
			metric: this.metric,
			cost_function: this.cost_function,
			regularization: this.regularization,
			regularization_ratio: this.regularization_ratio,
			net: this.net
		};

		// localStorage.setItem('neural_model_IA', JSON.stringify(obj));

		if (this.modelo_entrenado) {
			console.log('Guardando modelo entrenado');
			await this.model.save('indexeddb://model_IA').then((response) => {
				console.log(response);
			});
		}
	}

	async cargarModelo() {
		this.model = await tf.loadLayersModel('indexeddb://model_IA').then((response) => {
			console.log(response);
		});
	}
	// cvChanged(e) {
	// 	this.sumate.cv = e.target.files[0] !== undefined ? e.target.files[0] : undefined;
	// 	if(this.sumate.cv !== undefined)
	// 	  // en this.sumate.cv.name tenemos el nombre
	// 	else
	// 	  // aca no hay archivo cargado
	//   }

	async recuperarModelo() {
		let object = localStorage.getItem('neural_model_IA');
		let datos = JSON.parse(object);
		this.learning_ratio = datos.learning_ratio;
		this.epochs = datos.epochs;
		this.batch_check = datos.batch_check;
		this.batch_size = datos.batch_size;
		this.metric = datos.metric;
		this.cost_function = datos.cost_function;
		this.regularization = datos.regularization;
		this.regularization_ratio = datos.regularization_ratio;
		this.net = datos.net;

		this.model = await tf.loadLayersModel('indexeddb://model_IA');
	}

	armarStringCapa(tipo, units, activacion, ratio) {
		if (tipo === 'dense') {
			return (
				'\tmodel.add( tf.layers.' + tipo + '( { units: ' + units + ", activation: '" + activacion + "' } ) );"
			);
		} else {
			return '\tmodel.add( tf.layers.' + tipo + '( ' + ratio + ' ) );';
		}
	}

	actualizarCodigo1() {
		this.model_code = [];
		this.model_code.push('createModel() {');
		this.model_code.push('\tmodel = tf.sequential();');
		this.model_code.push('\tmodel.add( tf.layers.flatten( { inputShape: [ 28, 28, 1] } ) );');
		for (let i = 1; i < this.net.length; i++) {
			this.model_code.push(
				this.armarStringCapa(
					this.net[i].type.toLowerCase(),
					this.net[i].units.toString().toLowerCase(),
					this.net[i].activation.toLowerCase(),
					this.net[i].ratio.toString()
				)
			);
		}
		this.model_code.push('\treturn model;');
		this.model_code.push('}');
	}

	actualizarCodigo() {
		this.actualizarCodigo2();
		this.actualizarCodigo3();
	}

	actualizarCodigo2() {
		this.compile_code = [];
		this.compile_code.push('model.compile( {');
		this.compile_code.push('\toptimizer: tf.train.sgd(' + this.learning_ratio + '),');
		this.compile_code.splice(2, 1, "\tloss: '" + this.cost_function + "',");
		this.compile_code.splice(3, 1, "\tmetrics: [ '" + this.metric + "' ]");
		this.compile_code.push('} );');
	}
	actualizarCodigo3() {
		this.train_code = [];
		this.train_code.push('train(){');
		if (!this.batch_check) this.train_code.splice(1, 1, '\tbatchSize = ' + 32 + ';');
		else this.train_code.splice(1, 1, '\tbatchSize = ' + this.batch_size + ';');
		this.train_code.splice(2, 1, '\ttrainEpochs = ' + this.epochs + ';');
		this.train_code.splice(3, 1, '\ttrainData = getTrainData();');
		this.train_code.splice(4, 1, '\ttestData = getTestData();');
		this.train_code.splice(5, 1, '\tmodel.fit( trainData.images, trainData.labels, {');
		this.train_code.splice(6, 1, '\t\tepochs: trainEpochs,');
		this.train_code.splice(7, 1, '\t\tbatchSize: batchSize,');
		this.train_code.splice(8, 1, '\t\tcallbacks: { onBatchEnd }');
		this.train_code.splice(9, 1, '\t} );');
		this.train_code.splice(10, 1, '\ttestResult = model.evaluate( testData.images, testData.labels );');
		this.train_code.splice(11, 1, '\ttestAccuracyPercent = testResult[ 1 ].dataSync()[ 0 ] * 100;');
		this.train_code.splice(12, 1, '\ttestAccuracy = parseFloat( testAccuracyPercent.toFixed(2) );');
		this.train_code.splice(13, 1, '}');
		this.train_code.splice(14, 1, '\n');
		this.train_code.splice(15, 1, 'onBatchEnd( batch, logs ){');
		this.train_code.splice(16, 1, "\tconsole.log( 'Accuracy', logs.acc );");
		this.train_code.splice(17, 1, '}');
	}
	// //de la red.
	createDenseModel() {
		const model = tf.sequential();
		model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));
		model.add(tf.layers.dense({ units: 42, activation: 'relu' }));
		model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
		return model;
	}

	createModel() {
		const model = tf.sequential();

		model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));

		for (let i = 1; i < this.net.length; i++) {
			let layer = this.net[i];
			let tipo = this.net[i].type;
			if (tipo === 'Dense') {
				let act = layer.activation.toLowerCase();
				model.add(tf.layers.dense({ units: layer.units, activation: act }));
			} else if (tipo === 'Dropout') {
				model.add(tf.layers.dropout(layer.ratio));
			}
		}

		model.summary();
		// console.log(model.countParams());
		return model;
	}

	entrenar() {
		console.log('Comienzo entrenamiento');
		this.entrenando = true;
		this.resultados = true;

		this.model = this.createModel();
		this.hayResultados = true;

		this.train().then(() => {
			console.log(this.res_obtenidos);
			this.entrenando = false;
			// let y_pred = this.model.predict(this.data.xs);
			// this.predictions = Array.from(y_pred.argMax(1).dataSync());

			this.sliceEjemplos = this.sliceEjemplos - 10;
			this.cargarEjemplos(0);
			// this.test_data = this.getTestData(10);
			// let data = this.getTestData(20);
			// let y_pred = this.model.predict(this.test_data.xs);
			// const labels = Array.from(this.test_data.labels.argMax(1).dataSync());
			// const predictions = Array.from(y_pred.argMax(1).dataSync());

			// this.prediction_labels = labels;
			// this.prediction_nuevas = predictions;
			// this.ejemplos_cargados = true;
			// this.testExamplesLength = this.test_data.xs.shape[0];

			// const labels = Array.from(data.labels.argMax(1).dataSync());

			// this.prediction_labels = labels;
			// this.prediction_nuevas = predictions;
			// const testExamples = data.xs.shape[0];

			// for (let i = 0; i < testExamples; i++) {
			// 	const image = data.xs.slice([ i, 0 ], [ 1, data.xs.shape[1] ]);

			// 	let canvas = document.getElementById('canvas' + i);

			// 	this.draw(image.flatten(), canvas);
			// }
		});

		//this.load().then(() => {
		//});
	}

	draw(image, canvas) {
		const [ width, height ] = [ 28, 28 ];
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext('2d');
		const imageData = new ImageData(width, height);
		const data = image.dataSync();
		for (let i = 0; i < height * width; ++i) {
			const j = i * 4;
			imageData.data[j + 0] = data[i] * 255;
			imageData.data[j + 1] = data[i] * 255;
			imageData.data[j + 2] = data[i] * 255;
			imageData.data[j + 3] = 255;
		}

		ctx.putImageData(imageData, 0, 0);
	}

	// draw3(testExamples) {
	// 	for (let i = 0; i < testExamples; i++) {
	// 		const im = this.test_data.xs.slice([ i, 0 ], [ 1, this.test_data.xs.shape[1] ]);

	// 		let canvas = <HTMLCanvasElement>document.createElement('canvas');
	// 		// let canvas = document.createElement('canvas');
	// 		// document.getElementById('div_test_canvas').appendChild(canvas);
	// 		//this.draw(image.flatten(), canvas);
	// 		let image = im.flatten();
	// 		const [ width, height ] = [ 28, 28 ];
	// 		canvas.width = width;
	// 		canvas.height = height;
	// 		const ctx = canvas.getContext('2d');
	// 		const imageData = new ImageData(width, height);
	// 		const data = image.dataSync();
	// 		for (let i = 0; i < height * width; ++i) {
	// 			const j = i * 4;
	// 			imageData.data[j + 0] = data[i] * 255;
	// 			imageData.data[j + 1] = data[i] * 255;
	// 			imageData.data[j + 2] = data[i] * 255;
	// 			imageData.data[j + 3] = 255;
	// 		}
	// 		ctx.putImageData(imageData, 0, 0);

	// 		this.canvases.push(canvas);
	// 	}
	// }

	async train() {
		//const optimizer = 'rmsprop';

		//optimizer: tf.train.adam(0.001),
		// this.test_accuracy = 'Calculando...';
		// this.trainset_accuracy = 'Calculando...';
		//console.log(this.model !== undefined && this.model !== null);
		this.model.compile({
			optimizer: tf.train.sgd(this.learning_ratio),
			loss: 'categoricalCrossentropy',
			metrics: [ 'accuracy' ]
		});

		this.progreso = 0;

		const batchSize = this.batch_check ? this.batch_size : 32;

		// Leave out the last 15% of the training data for validation, to monitor
		// overfitting during training.
		const validationSplit = 0.15;

		// Get number of training epochs from the UI.
		const trainEpochs = this.epochs;

		// We'll keep a buffer of loss and accuracy values over time.
		let trainBatchCount = 0;

		const trainData = this.getTrainData();
		const testData = this.getTestData(null);

		const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;

		let valAcc;
		let trainsetAcc;

		await this.model.fit(trainData.xs, trainData.labels, {
			batchSize,
			validationSplit,
			epochs: trainEpochs,
			callbacks: {
				onBatchEnd: async (batch, logs) => {
					trainBatchCount++;
					this.progreso = Math.floor(trainBatchCount / totalNumBatches * 100);
					console.log(logs);
					if (this.res_obtenidos[this.epochActual + 1] === undefined) {
						this.res_obtenidos[this.epochActual + 1] = {
							loss: logs.loss.toFixed(4),
							acc: logs.acc.toFixed(4),
							val_loss: '-',
							val_acc: '-'
						};
					} else {
						this.res_obtenidos[this.epochActual + 1]['loss'] = logs.loss.toFixed(4);
						this.res_obtenidos[this.epochActual + 1]['acc'] = logs.acc.toFixed(4);
					}

					// this.validation_accuracy = logs.val_acc.toFixed(2) + '%';
					this.trainset_accuracy = logs.acc.toFixed(2) + '%';
					console.log(
						`Training... (` + `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` + ` complete).`
					);
				},
				onEpochEnd: async (epoch, logs) => {
					// valAcc = logs.val_acc;
					// trainsetAcc = logs.acc;
					this.validation_accuracy = logs.val_acc.toFixed(2) + '%';
					this.trainset_accuracy = logs.acc.toFixed(2) + '%';

					this.res_obtenidos[this.epochActual + 1]['loss'] = logs.loss.toFixed(4);
					this.res_obtenidos[this.epochActual + 1]['acc'] = logs.acc.toFixed(4);
					this.res_obtenidos[this.epochActual + 1]['val_loss'] = logs.val_loss.toFixed(4);
					this.res_obtenidos[this.epochActual + 1]['val_acc'] = logs.val_acc.toFixed(4);

					this.dataLossChart.labels.push('0');
					this.dataLossChart.datasets[0].data.push(logs.loss.toFixed(4));
					this.dataLossChart.datasets[1].data.push(logs.val_loss.toFixed(4));
					this.dataAccChart.labels.push('0');
					this.dataAccChart.datasets[0].data.push(logs.acc.toFixed(4));
					this.dataAccChart.datasets[1].data.push(logs.val_acc.toFixed(4));

					this.lossChart.update();
					this.accuracyChart.update();

					this.epochActual = epoch + 1;
				}
			}
		});

		const testResult = this.model.evaluate(testData.xs, testData.labels);
		const testAccPercent = testResult[1].dataSync()[0] * 100;
		// const finalValAccPercent = valAcc * 100;
		// const finalTrainsetAccPercent = trainsetAcc * 100;
		// this.validation_accuracy = parseFloat(finalValAccPercent.toFixed(2));
		// this.test_accuracy = testAccPercent.toFixed(2) + ' %';
		// this.trainset_accuracy = finalTrainsetAccPercent.toFixed(2) + '%';
		// console.log(
		// 	`Final train set accuracy: ${finalTrainsetAccPercent.toFixed(1)}%; ` +
		// 		`Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
		// 		`Final test accuracy: ${testAccPercent.toFixed(1)}%` +
		// 		`Test result: ${testResult}`
		// );
		this.modelo_entrenado = true;
	}

	async load() {
		const img = new Image();
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');

		const imgRequest = new Promise((resolve, reject) => {
			img.crossOrigin = '';
			img.onload = () => {
				img.width = img.naturalWidth;
				img.height = img.naturalHeight;

				const datasetBytesBuffer = new ArrayBuffer(this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE * 4);

				const chunkSize = 5000;
				canvas.width = img.width;
				canvas.height = chunkSize;

				for (let i = 0; i < this.NUM_DATASET_ELEMENTS / chunkSize; i++) {
					const datasetBytesView = new Float32Array(
						datasetBytesBuffer,
						i * this.IMAGE_SIZE * chunkSize * 4,
						this.IMAGE_SIZE * chunkSize
					);
					ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

					const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

					for (let j = 0; j < imageData.data.length / 4; j++) {
						// All channels hold an equal value since the image is grayscale, so
						// just read the red channel.
						datasetBytesView[j] = imageData.data[j * 4] / 255;
					}
				}
				this.datasetImages = new Float32Array(datasetBytesBuffer);

				resolve();
			};
			img.src = this.MNIST_IMAGES_SPRITE_PATH;
		});

		//console.log(this.datasetImages);

		const labelsRequest = fetch(this.MNIST_LABELS_PATH);
		const [ imgResponse, labelsResponse ] = await Promise.all([ imgRequest, labelsRequest ]);

		this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

		// Slice the the images and labels into train and test sets.
		this.trainImages = this.datasetImages.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);

		//console.log(this.trainImages);
		this.testImages = this.datasetImages.slice(this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
		this.trainLabels = this.datasetLabels.slice(0, this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
		this.testLabels = this.datasetLabels.slice(this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
	}

	getTrainData() {
		const xs = tf.tensor4d(this.trainImages, [
			this.trainImages.length / this.IMAGE_SIZE,
			this.IMAGE_H,
			this.IMAGE_W,
			1
		]);
		const labels = tf.tensor2d(this.trainLabels, [ this.trainLabels.length / this.NUM_CLASSES, this.NUM_CLASSES ]);

		return { xs, labels };
	}

	getTestData(numExamples) {
		let xs = tf.tensor4d(this.testImages, [
			this.testImages.length / this.IMAGE_SIZE,
			this.IMAGE_H,
			this.IMAGE_W,
			1
		]);
		let labels = tf.tensor2d(this.testLabels, [ this.testLabels.length / this.NUM_CLASSES, this.NUM_CLASSES ]);
		//console.log(this.sliceEjemplos);

		if (numExamples != null) {
			//const r = Math.floor(Math.random() * (this.testImages.length / this.IMAGE_SIZE + 1)) - numExamples;

			xs = xs.slice([ this.sliceEjemplos, 0, 0, 0 ], [ numExamples, this.IMAGE_H, this.IMAGE_W, 1 ]);
			labels = labels.slice([ this.sliceEjemplos, 0 ], [ numExamples, this.NUM_CLASSES ]);
			this.sliceEjemplos = this.sliceEjemplos + numExamples;
		}

		return { xs, labels };
	}

	// metodos del canvas para dibujar

	async predict() {
		let canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
		let preview = <HTMLCanvasElement>document.getElementById('preview-canvas');

		let img = tf.browser.fromPixels(canvas, 4);
		let resized = this.cropImage(img, canvas.width);
		tf.browser.toPixels(resized, preview);

		let x_data = tf.cast(resized.reshape([ 1, 28, 28, 1 ]), 'float32');

		let y_pred = this.model.predict(x_data);

		this.predictions = Array.from(y_pred.dataSync());
		console.log(this.predictions);
		console.log(y_pred.argMax(1).dataSync());
		this.prediction = this.predictions.indexOf(Math.max(...this.predictions));

		// this.prediction = Array.from(y_pred.argMax(1).dataSync());
		// console.log(this.prediction);
	}

	clearCanvas() {
		let canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
		canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
	}

	cropImage(img, width = 140) {
		img = img.slice([ 0, 0, 3 ]);
		var mask_x = tf.greater(img.sum(0), 0).reshape([ -1 ]);
		var mask_y = tf.greater(img.sum(1), 0).reshape([ -1 ]);
		var st = tf.stack([ mask_x, mask_y ]);
		var v1 = tf.topk(st);
		var v2 = tf.topk(st.reverse());

		var [ x1, y1 ] = v1.indices.dataSync();
		var [ y2, x2 ] = v2.indices.dataSync();
		y2 = width - y2 - 1;
		x2 = width - x2 - 1;
		var crop_w = x2 - x1;
		var crop_h = y2 - y1;

		if (crop_w > crop_h) {
			y1 -= (crop_w - crop_h) / 2;
			crop_h = crop_w;
		}
		if (crop_h > crop_w) {
			x1 -= (crop_h - crop_w) / 2;
			crop_w = crop_h;
		}

		img = img.slice([ y1, x1 ], [ crop_h, crop_w ]);
		img = img.pad([ [ 6, 6 ], [ 6, 6 ], [ 0, 0 ] ]);
		var resized = tf.image.resizeNearestNeighbor(img, [ 28, 28 ]);

		for (let i = 0; i < 28 * 28; i++) {
			resized[i] = 255 - resized[i];
		}
		return resized;
	}

	startPosition() {
		this.painting = true;
		console.log('hola');
	}

	finishedPosition() {
		this.painting = false;

		if (this.canvas == undefined) {
			this.canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
			this.cx = this.canvas.getContext('2d');
		}
		this.cx.beginPath();
	}

	mouseMove(e) {
		console.log(e);
		if (!this.painting) return;
		if (this.canvas == undefined) {
			this.canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
			this.cx = this.canvas.getContext('2d');
		}

		const rect = this.canvas.getBoundingClientRect();

		this.cx.lineWidth = 16;
		this.cx.lineCap = 'round';
		this.cx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
		this.cx.stroke();
		this.cx.beginPath();
		this.cx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
	}

	newLayer() {
		this.net.splice(this.net.length - 1, 0, new Layer('Dense', 0, 'ReLU', null, 0));
		this.cant_capas++;
		console.log(this.net);
		this.actualizarCodigo1();
	}

	removeLayer(index) {
		this.net.splice(index, 1);
		this.cant_capas--;
		this.actualizarCodigo1();
	}

	moverCapa(index, dir) {
		let item = this.net[index];
		if (dir > 0) {
			this.net[index] = this.net[index + 1];
			this.net[index + 1] = item;
		} else {
			this.net[index] = this.net[index - 1];
			this.net[index - 1] = item;
		}
		this.actualizarCodigo1();
	}

	onChange(value, field, index) {
		if (field !== 'units' && field !== 'ratio') {
			this.net[index][field] = value;
		} else {
			if (field === 'units') this.net[index][field] = parseInt(value);
			else if (field === 'ratio') {
				this.net[index][field] = parseFloat(value);
			}
		}

		// console.log(this.net);

		this.actualizarCodigo1();
	}

	unsorted() {
		//Ordenador para la tabla del proceso de entrenamiento. (vacío a propósito)
	}

	descartes() {
		// this.canvas = <HTMLCanvasElement>document.getElementById('predict-canvas');
		// this.cx = this.canvas.getContext('2d');
		// this.canvas.addEventListener('mousedown', this.startPosition);
		// this.canvas.addEventListener('mouseup', this.finishedPosition);
		// this.canvas.addEventListener('mousemove', this.mouseMove);
	}
}
