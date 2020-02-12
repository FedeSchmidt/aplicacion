import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Layer } from './../models/layer';
import Chart from 'chart.js';
import { JsonPipe } from '@angular/common';
import { VirtualTimeScheduler } from 'rxjs';

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

	datasetImages: Float32Array;
	datasetLabels: Uint8Array;
	trainImages: any;
	testImages: any;
	trainLabels: any;
	testLabels: any;

	mostrar_info = false;
	error = false;
	mensaje_error = "";

	//codigo equivalente
	mostrar_codigo = true;
	model_code = [];
	train_code = [];
	compile_code = [];

	//ejemplos
	data;
	sliceEjemplos = 0;
	modelo_entrenado = false;
	prediction_labels;
	predictions: any;
	imagenes_cargadas = false;

	//Entrenamiento
	res_obtenidos = {};
	entrenando = false;
	resultados = false;
	hayResultados = false;

	//Parámetros y red
	net = [];
	max_capas = 12;
	cant_capas = 0;
	tipo_red = 'Simple';
	model: any;
	validation_accuracy = '---';
	trainset_accuracy = '---';

	// parámetros generales
	learning_ratio = 0.15;
	epochs = 10;
	epochActual = 0;
	batch_size = 120;
	metricas = [ 'accuracy' ];
	metric = 'accuracy';
	loss_functions = [ 'categoricalCrossentropy', 'mean_squared_error' ];
	cost_function = 'categoricalCrossentropy';

	//Nodos
	tipos_capas = [ 'Dense', 'Convolutional', 'Pooling', 'Dropout', 'Flatten' ];
	activations = [ 'ReLU', 'Sigmoid', 'Linear', 'Softmax' ];
	windows = [ '2x2', '3x3', '5x5' ];

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
		//Arma la red de inicio
		this.actualizarRedBase();

		//Carga los gráficos de entrenamiento
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

		//Carga las imágenes y las separa en conjunto de entrenamiento y test con el load.
		//Después carga 10 ejemplos del conjunto de test para mostrarlos.
		this.load().then(() => {
			this.cargarEjemplos(0);
		});
	}

	// armarRedBase() {
	// 	this.net.push(new Layer('Flatten', 784, '-', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '3x3'));
	// 	this.net.push(new Layer('Dense', 42, 'ReLU', null, 0, '3x3'));
	// 	this.net.push(new Layer('Dense', 10, 'Softmax', null, 0, '3x3'));
	// 	this.cant_capas = this.net.length;

	// 	this.actualizarCodigo1();
	// 	this.actualizarCodigo();
	// }

	actualizarRedBase() {
		//Arma una red modelo según el tipo que elige el usuario
		//Actualiza el código que se ve abajo.
		this.net = [];
		if (this.tipo_red === 'Simple') {
			this.net.push(new Layer('Flatten', 784, '-', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '3x3'));
			this.net.push(new Layer('Dense', 42, 'ReLU', null, 0, '3x3'));
			this.net.push(new Layer('Dense', 10, 'Softmax', null, 0, '3x3'));
		} else {
			this.net.push(new Layer('Convolutional', 32, 'ReLU', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '3x3'));
			this.net.push(new Layer('Convolutional', 32, 'ReLU', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '3x3'));
			this.net.push(new Layer('Pooling', 32, '-', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '2x2'));
			this.net.push(new Layer('Flatten', 32, '-', [ this.IMAGE_H, this.IMAGE_W, 1 ], 0, '2x2'));
			this.net.push(new Layer('Dense', 10, 'Softmax', null, 0, '3x3'));
		}
		this.cant_capas = this.net.length;

		this.actualizarCodigo1();
		this.actualizarCodigo();
	}

	showCodigos() {
		//Para mostrar o esconder la sección de código de abajo.
		this.mostrar_codigo = !this.mostrar_codigo;
	}

	cargarEjemplos(diff) {
		if (diff < 0) {
			this.sliceEjemplos = this.sliceEjemplos - 20;
		}

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

	guardarModelo() {
		//Armo un objeto con los parámetros y la estructura de la red.
		//Lo descarga con un link ficticio
		let obj = {
			tipo_red: this.tipo_red,
			learning_ratio: this.learning_ratio,
			epochs: this.epochs,
			metric: this.metric,
			cost_function: this.cost_function,
			batch_size: this.batch_size,
			net: this.net
		};

		let name = 'model.json';
		const type = name.split('.').pop();

		const a = document.createElement('a');
		a.href = URL.createObjectURL(
			new Blob([ JSON.stringify(obj) ], { type: `text/${type === 'txt' ? 'plain' : type}` })
		);
		a.download = name;
		a.click();
	}

	cvChanged(event) {
		//Cargo un archivo  y actualizo los parámetros de red y la estructura
		//Actualiza el código de abajo y el valor del input se resetea para que pueda usarlo de nuevo, si no no anda.
		let file_to_read = event.target.files[0];
		var fileread = new FileReader();

		fileread.onloadend = (e) => {
			let data = JSON.parse(fileread.result as string);

			this.tipo_red = data.tipo_red;
			this.learning_ratio = data.learning_ratio;
			this.epochs = data.epochs;
			this.batch_size = data.batch_size;
			this.metric = data.metric;
			this.cost_function = data.cost_function;
			this.net = data.net;
			this.actualizarCodigo1();
			this.actualizarCodigo();
			event.srcElement.value = '';
		};
		fileread.readAsText(file_to_read);
	}

	armarStringCapa(tipo, units, activacion, ratio, window) {
		//Arma el string de una capa, es solo manejo de string con la estructura de la red.
		let w = window.charAt(0);
		if (tipo === 'dense')
			return (
				'\tmodel.add( tf.layers.' + tipo + '( { units: ' + units + ", activation: '" + activacion + "' } ) );"
			);
		else if (tipo === 'dropout') return '\tmodel.add( tf.layers.' + tipo + '( ' + ratio + ' ) );';
		else if (tipo === 'flatten') return '\tmodel.add( tf.layers.' + tipo + '() );';
		else if (tipo === 'pooling') return '\tmodel.add( tf.layers.' + tipo + '((' + w + ', ' + w + '));';
		else
			return (
				'\tmodel.add( tf.layers.conv2D( { ' +
				units +
				', (' +
				w +
				', ' +
				w +
				"), activation: '" +
				activacion +
				"' } ) );"
			);
	}

	actualizarCodigo1() {
		//idem anterior.
		this.model_code = [];
		this.model_code.push('createModel() {');
		this.model_code.push('\tmodel = tf.sequential();');
		if (this.tipo_red === 'Simple') {
			this.model_code.push('\tmodel.add( tf.layers.flatten( { inputShape: [ 28, 28, 1 ] } ) );');

			for (let i = 1; i < this.net.length; i++) {
				this.model_code.push(
					this.armarStringCapa(
						this.net[i].type.toLowerCase(),
						this.net[i].units.toString().toLowerCase(),
						this.net[i].activation.toLowerCase(),
						this.net[i].ratio.toString(),
						this.net[i].window.toLowerCase()
					)
				);
			}
		} else {
			let wf = this.net[0].window.charAt(0);

			this.model_code.push(
				'\tmodel.add( tf.layers.conv2D( { ' +
					this.net[0].units +
					', (' +
					wf +
					', ' +
					wf +
					"), activation: '" +
					this.net[0].activation +
					"', inputShape: [ 28, 28, 1 ] } ) );"
			);

			for (let i = 1; i < this.net.length; i++) {
				this.model_code.push(
					this.armarStringCapa(
						this.net[i].type.toLowerCase(),
						this.net[i].units.toString().toLowerCase(),
						this.net[i].activation.toLowerCase(),
						this.net[i].ratio.toString(),
						this.net[i].window.toLowerCase()
					)
				);
			}
		}
		this.model_code.push('\treturn model;');
		this.model_code.push('}');
	}

	actualizarCodigo() {
		this.actualizarCodigo2();
		this.actualizarCodigo3();
	}

	actualizarCodigo2() {
		//idem anterior
		this.compile_code = [];
		this.compile_code.push('model.compile( {');
		this.compile_code.push('\toptimizer: tf.train.sgd(' + this.learning_ratio + '),');
		this.compile_code.splice(2, 1, "\tloss: '" + this.cost_function + "',");
		this.compile_code.splice(3, 1, "\tmetrics: [ '" + this.metric + "' ]");
		this.compile_code.push('} );');
	}
	actualizarCodigo3() {
		//idem anterior
		this.train_code = [];
		this.train_code.push('train(){');
		this.train_code.splice(1, 1, '\tbatchSize = ' + this.batch_size + ';');
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

	createDenseModel() {
		//Crea red estática, no depende de la tabla que maneja el usuario. No se usa este método. Ver createModel
		const model = tf.sequential();
		model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));
		model.add(tf.layers.dense({ units: 42, activation: 'relu' }));
		model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
		return model;
	}

	createModel() {
		//Crea estructura según la tabla
		//La primera capa se agrega a mano, luego un for sobre el arreglo net va agregando las capas
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
		//model.summary();
		return model;
	}

	createConvModel() {
		//Igual al anterior pero para una red convolucional. Separado para modular.
		const model = tf.sequential();
		let w = parseInt(this.net[0].window.charAt(0));
		model.add(
			tf.layers.conv2d({
				filters: this.net[0].units,
				kernelSize: w,
				inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ]
			})
		);
		for (let i = 1; i < this.net.length; i++) {
			let layer = this.net[i];
			let tipo = this.net[i].type;
			if (tipo === 'Dense') {
				let act = layer.activation.toLowerCase();
				model.add(tf.layers.dense({ units: layer.units, activation: act }));
			} else if (tipo === 'Dropout') {
				model.add(tf.layers.dropout(layer.ratio));
			} else if (tipo === 'Convolutional') {
				let act = layer.activation.toLowerCase();
				let w = parseInt(layer.window.charAt(0));
				model.add(tf.layers.conv2d({ filters: layer.units, kernelSize: w, activation: act }));
			} else if (tipo === 'Flatten') {
				model.add(tf.layers.flatten());
			} else {
				let w = parseInt(layer.window.charAt(0));
				model.add(tf.layers.maxPooling2d({ poolSize: w }));
			}
		}
		//model.summary();
		return model;
	}

	test() {
		// if (this.tipo_red === 'Simple') this.createModel();
		// else this.createConvModel();
		//console.log(this.net);
		this.chequearEstructura();
		//this.mostrar_info = !this.mostrar_info;
	}

	closeAlert(){
		this.error = !this.error;
	}

	entrenar() {
		if(this.chequearEstructura()){

			this.entrenando = true;
			this.resultados = true;
			
			//Reinicia la tabla de entrenamiento y los gráficos (para permitir dos entrenamientos seguidos sin recargar)
			this.epochActual = 0;
			this.res_obtenidos = {};
			this.dataLossChart.labels = [];
			this.dataLossChart.datasets[0].data = [];
			this.dataLossChart.datasets[1].data = [];
			this.dataAccChart.labels = [];
			this.dataAccChart.datasets[0].data = [];
			this.dataAccChart.datasets[1].data = [];
			
			//Crea modelo, segun sea simple o convolucional.
			if (this.tipo_red === 'Simple') this.model = this.createModel();
			else this.model = this.createConvModel();
			
			//Variable para que se muestren los porcentajes y los gráficos con opacidad 1.
			this.hayResultados = true;
			
			//Entrena y luego predice sobre los ejemplos que se veían en pantalla.
			this.train().then(() => {
				this.entrenando = false;
				this.sliceEjemplos = this.sliceEjemplos - 10;
				this.cargarEjemplos(0);
			});
		}
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

	async train() {
		let loss;

		//Elige la función de costo del modelo
		if (this.cost_function === 'categoricalCrossentropy') loss = 'categoricalCrossentropy';
		else loss = tf.losses.meanSquaredError;

		//Compilación
		this.model.compile({
			optimizer: tf.train.sgd(this.learning_ratio),
			loss: loss,
			metrics: 'accuracy'
		});
		const batchSize = this.batch_size;

		// Leave out the last 15% of the training data for validation, to monitor overfitting during training.
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
					this.trainset_accuracy = parseFloat(logs.acc.toFixed(4)) * 100 + '%';
					// console.log(
					// 	`Training... (` + `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` + ` complete).`
					// );
				},
				onEpochEnd: async (epoch, logs) => {
					this.validation_accuracy = parseFloat(logs.val_acc.toFixed(4)) * 100 + '%';
					this.trainset_accuracy = parseFloat(logs.acc.toFixed(4)) * 100 + '%';

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

		// const testResult = this.model.evaluate(testData.xs, testData.labels);
		// const testAccPercent = testResult[1].dataSync()[0] * 100;

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

		const labelsRequest = fetch(this.MNIST_LABELS_PATH);
		const [ imgResponse, labelsResponse ] = await Promise.all([ imgRequest, labelsRequest ]);

		this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

		// Slice the the images and labels into train and test sets.
		this.trainImages = this.datasetImages.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
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

		if (numExamples != null) {
			let a = this.sliceEjemplos;
			if (this.sliceEjemplos < 0) {
				a = this.NUM_TEST_ELEMENTS + this.sliceEjemplos;
			}

			xs = xs.slice([ a, 0, 0, 0 ], [ numExamples, this.IMAGE_H, this.IMAGE_W, 1 ]);
			labels = labels.slice([ a, 0 ], [ numExamples, this.NUM_CLASSES ]);
			this.sliceEjemplos = this.sliceEjemplos + numExamples;
		}

		return { xs, labels };
	}

	newLayer() {
		this.net.splice(this.net.length - 1, 0, new Layer('Dense', 0, 'ReLU', null, 0, '3x3'));
		this.cant_capas++;
		this.actualizarCodigo1();
	}

	removeLayer(index) {
		this.net.splice(index, 1);
		this.cant_capas--;
		this.actualizarCodigo1();
	}

	moverCapa(index, dir) {
		if (dir < 0 && index == 1) {
			//No se puede mover
		} else {
			if (dir > 0 && index == this.net.length - 2) {
				//No se puede mover
			} else {
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
		}
	}

	onChange(value, field, index) {
		console.log(value);
		if (field === 'units') this.net[index][field] = parseInt(value);
		else if (field === 'ratio') this.net[index][field] = parseFloat(value);
		else if (field === 'type') {
			this.net[index][field] = value;
			//Si cambié a una capa pooling tengo que setear window
			if (value === 'Pooling') {
				this.net[index]['window'] = '2x2';
			} else {
				//Si cambié a una capa convolucional, tengo que setear activation, units y window
				if (value === 'Convolutional') {
					this.net[index]['activation'] = 'ReLU';
					this.net[index]['units'] = 32;
					this.net[index]['window'] = '3x3';
				} else {
					//Si cambié a una capa Flatten no importa, no seteo nada
					//Si cambié a una capa Dropout tengo que setear ratio
					if (value === 'Dropout') this.net[index]['ratio'] = 0;
					else {
						//Si cambié a una capa Dense tengo que setear units y activation
						if (value === 'Dense') {
							this.net[index]['units'] = 0;
							this.net[index]['activation'] = 'ReLU';
						}
					}
				}
			}
		}

		this.actualizarCodigo1();
	}

	listaValidos(i) {
		let ant = this.net[i - 1].type;
		if (ant == 'Convolutional') return [ 'Convolutional', 'Pooling', 'Dropout', 'Flatten' ];
		else if (ant == 'Dense') return [ 'Dense', 'Dropout' ];
		else if (ant == 'Flatten') return [ 'Dense' ];
		else if (ant == 'Pooling') return [ 'Convolutional', 'Pooling', 'Dropout', 'Flatten' ];
		else {
			//ant era dropout
			let ant2 = this.net[i - 2].type;
			if (ant2 == 'Dense') return [ 'Dense' ];
			else {
				//ant2 es convolutional o pooling
				return [ 'Convolutional', 'Pooling', 'Flatten' ];
			}
		}
	}

	chequearEstructura(){
		let salida = true;
		if(this.tipo_red === 'Convolucional'){
			for(let i = 1; i < this.net.length; i++){
				let validos = this.listaValidos(i);
				if(validos.indexOf(this.net[i]['type']) == -1){
					// console.log("error de estructura" + (i-1) + "," + (i));
					this.error = true;
					this.mensaje_error = "Error en la estructura. Conflicto entre las capas "+ (i-1) +" y "+i;
					salida = false;
				}
			}
		}
		return salida;
	}

	unsorted() {
		//Ordenador para la tabla del proceso de entrenamiento. (vacío a propósito)
	}
}
