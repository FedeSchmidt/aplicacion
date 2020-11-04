import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Layer } from './../models/layer';
import Chart from 'chart.js';
import { ToastrService } from 'ngx-toastr';

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
	// NUM_TRAIN_ELEMENTS = 55000;
	porc_examples_test = 0.2;
	porc_examples_train = 0;
	porc_examples_val = 0;
	EXAMPLES_TRAIN = 0
	EXAMPLES_VAL = 0
	ratio_val = 0.2;
	NUM_TEST_ELEMENTS = Math.floor(this.NUM_DATASET_ELEMENTS * this.porc_examples_test);
	NUM_TRAIN_ELEMENTS = this.NUM_DATASET_ELEMENTS - this.NUM_TEST_ELEMENTS;
	IMAGE_H = 28;
	IMAGE_W = 28;
	posible_cargar = false;

	datasetImages: Float32Array;
	datasetLabels: Uint8Array;
	trainImages: any;
	testImages: any;
	trainLabels: any;
	testLabels: any;

	mostrar_info = false;
	error = false;
	mensaje_error = '';

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
	grados_conf: any;
	y_pred: any;
	barChartData: Array<number>;
	imagenes_cargadas = false;

	//Entrenamiento
	res_obtenidos: any = {};
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
	learning_ratio = 0.1;
	epochs = 10;
	epochActual = 0;
	batch_size = 120;
	metricas = [ 'accuracy' ];
	metric = 'accuracy';
	loss_functions = [ 'categorical_crossentropy', 'mean_squared_error' ];
	optimizers = [ 'SGD', 'Adam' ];
	cost_function = 'categorical_crossentropy';
	optimizer = 'SGD';

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

	dataBarChart = {
		labels: [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ],
		datasets: [
			{
				barPercentage: 0.7,
				data: [ 0.5, 0.5, 0.4, 0, 1 ],
				backgroundColor: 'rgba(188, 145, 128, 0.5)',
				borderColor: 'rgba(188, 145, 128, 1)',
				borderWidth: 1
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

	optionsBarChart = {
		title: {
			display: false
		},
		legend: {
			display: false
		},
		scales: {
			xAxes: [
				{
					gridLines: {
						display: false
					}
				}
			],
			yAxes: [
				{
					ticks: {
						suggestedMin: 0,
						suggestedMax: 1
					}
				}
			]
		}
	};
	accuracyChart: any;
	lossChart: any;
	predictionChart: any;

	constructor(private toastr: ToastrService) {}

	ngOnInit() {
		this.porc_examples_train = Math.round((1 - this.porc_examples_test)* (1-this.ratio_val) * 100);
		this.porc_examples_val = Math.round((1 - this.porc_examples_test)* this.ratio_val * 100);

		this.EXAMPLES_TRAIN = Math.round(this.NUM_TRAIN_ELEMENTS * (1- this.ratio_val));
		this.EXAMPLES_VAL = Math.round(this.NUM_TRAIN_ELEMENTS * this.ratio_val);

		if (typeof(Storage) !== "undefined") {
			// LocalStorage disponible
			if(localStorage.getItem("network") != undefined)
				this.posible_cargar = true;
		}
		
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

		var canv = <HTMLCanvasElement>document.getElementById('bar-chart');
		var ctx3 = canv.getContext('2d');
		this.predictionChart = new Chart(ctx3, {
			type: 'bar',
			data: this.dataBarChart,
			options: this.optionsBarChart
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

	closeAlert() {
		//Cierra una alenta de error
		this.error = false;
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

		if (this.modelo_entrenado) {
			this.y_pred = this.model.predict(this.data.xs);
			this.predictions = Array.from(this.y_pred.argMax(1).dataSync());
			const conf = this.y_pred.max(1);
			const scalar = tf.scalar(100);
			this.grados_conf = Array.from(tf.mul(conf, scalar).dataSync());
			this.barChartData = Array.from(this.y_pred.dataSync());
		}

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
				if (document.getElementById('prediccion' + i) == undefined) {
					let prediccion = document.createElement('h6');
					prediccion.id = 'prediccion' + i;
					prediccion.innerHTML = 'Predicción: ' + this.predictions[i];
					let confianza = document.createElement('h6');
					confianza.id = 'conf' + i;
					if(this.grados_conf[i] < 100){
						confianza.innerHTML = 'Conf: ' + this.grados_conf[i].toFixed(2) + '%';
					}else{
						confianza.innerHTML = 'Conf: ' + this.grados_conf[i].toFixed(0) + '%';
					}
					div.appendChild(prediccion);
					div.appendChild(confianza);
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

				canvas.setAttribute('data-toggle', 'modal');
				canvas.setAttribute('data-target', '#modalBarChart');
				canvas.addEventListener('click', () => {
					let probs = [] as any;
					for (let j = 10 * i; j < 10 * i + 10; j++) {
						let p = parseFloat(this.barChartData[j].toFixed(3));
						probs.push(p);
					}
					this.dataBarChart.datasets[0].data = probs;
					this.predictionChart.update();
				});
			}
		}
	}

	guardarModelo(){
		if (typeof(Storage) !== "undefined") {
			// LocalStorage disponible

			let obj = {
				tipo_red: this.tipo_red,
				learning_ratio: this.learning_ratio,
				epochs: this.epochs,
				metric: this.metric,
				cost_function: this.cost_function,
				batch_size: this.batch_size,
				optimizer: this.optimizer,
				porc_examples_test: this.porc_examples_test,
				net: this.net
			};
			localStorage.setItem("network", JSON.stringify(obj));

			this.posible_cargar = true;
			this.toastr.success('Modelo guardado en localStorage');

		} else {
			// LocalStorage no soportado en este navegador
			console.log("No se pudo guardar la información...");
			this.toastr.warning('No se pudo guardar la información');
		}		
	}

	cargarModelo(){
		if (typeof(Storage) !== "undefined") {
			// LocalStorage disponible
			const data = JSON.parse(localStorage.getItem("network"));

			this.tipo_red = data.tipo_red;
			this.learning_ratio = data.learning_ratio;
			this.epochs = data.epochs;
			this.batch_size = data.batch_size;
			this.metric = data.metric;
			this.cost_function = data.cost_function;
			this.optimizer = data.optimizer;
			this.porc_examples_test = data.porc_examples_test;
			this.net = data.net;
			this.actualizarCodigo1();
			this.actualizarCodigo();

			this.toastr.success('Modelo cargado de localStorage');
		} else {
			// LocalStorage no soportado en este navegador
			console.log("No se pudo cargar la información...");
			this.toastr.warning('No se pudo cargar la información');
		}
	}

	// async exportarModelo(){
	// 	if(this.modelo_entrenado){

	// 		await this.model.save('downloads://model_mnist');
	// 	}
	// }
	exportarModelo() {
		if (this.chequearEstructura()) {
			//Armo un objeto con los parámetros y la estructura de la red.
			//Lo descarga con un link ficticio
			let obj = {
				tipo_red: this.tipo_red,
				learning_ratio: this.learning_ratio,
				epochs: this.epochs,
				metric: this.metric,
				cost_function: this.cost_function,
				batch_size: this.batch_size,
				optimizer: this.optimizer,
				porc_examples_test: this.porc_examples_test,
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
			this.optimizer = data.optimizer;
			this.porc_examples_test = data.porc_examples_test;
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
			return 'model.add( tf.layers.' + tipo + '( units = ' + units + ", activation = '" + activacion + "'));";
		else if (tipo === 'dropout') return 'model.add( tf.layers.' + tipo + '( ' + ratio + ' ));';
		else if (tipo === 'flatten') return 'model.add( tf.layers.' + tipo + '());';
		else if (tipo === 'pooling') return 'model.add( tf.layers.' + tipo + '((' + w + ', ' + w + '));';
		else
			return (
				'model.add( tf.layers.conv2D( units = ' +
				units +
				', filter = (' +
				w +
				', ' +
				w +
				"), activation = '" +
				activacion +
				"'));"
			);
	}

	actualizarCodigo1() {
		//idem anterior.
		this.model_code = [];
		this.model_code.push('// Creamos la estructura de la RNA');
		this.model_code.push('model = tf.sequential();');
		if (this.tipo_red === 'Simple') {
			this.model_code.push('model.add( tf.layers.flatten( inputShape = (28, 28, 1)));');

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
				'model.add( tf.layers.conv2D( units = ' +
					this.net[0].units +
					', filter = (' +
					wf +
					', ' +
					wf +
					"), activation =  '" +
					this.net[0].activation.toLowerCase() +
					"', inputShape = (28, 28, 1)));"
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
	}

	actualizarCodigo() {
		this.actualizarCodigo2();
		this.actualizarCodigo3();
	}

	actualizarCodigo2() {
		//idem anterior
		this.compile_code = [];
		this.compile_code.push('// Compilamos el modelo');
		this.compile_code.splice(1, 0, 'model.compile( ');
		this.compile_code.splice(
			2,
			0,
			'\t\toptimizer =  tf.train.' + this.optimizer.toLowerCase() + '(' + this.learning_ratio + '),'
		);
		this.compile_code.splice(3, 0, "\t\tloss = '" + this.cost_function + "',");
		this.compile_code.splice(4, 0, "\t\tmetrics = [ '" + this.metric + "' ]");
		this.compile_code.push(');');
	}
	actualizarCodigo3() {
		//idem anterior
		this.train_code = [];
		this.train_code.push('// Entrenamiento');
		this.train_code.push('// Los datos son una colección de pares (ejemplo, label)');
		this.train_code.splice(2, 0, 'data = [ ... cuerpo de datos ... ]');
		this.train_code.splice(3, 0, 'batchSize = ' + this.batch_size + ';');
		this.train_code.splice(4, 0, 'trainEpochs = ' + this.epochs + ';');
		this.train_code.splice(5, 0, 'trainData, valData, testData = train_test_split(data);');
		this.train_code.splice(6, 0, 'model.fit( trainData.images, trainData.labels,');
		this.train_code.splice(7, 0, '\t\tepochs = trainEpochs,');
		this.train_code.splice(8, 0, '\t\tbatchSize = batchSize,');
		this.train_code.splice(9, 0, '\t\tvalidation_data = (valData.images, valData.labels)');
		this.train_code.splice(10, 0, ');');
		this.train_code.splice(11, 0, '// Predicciones');
		this.train_code.splice(12, 0, 'test_results = model.predict(testData.images);');
		this.train_code.splice(13, 0, 'predictions = tf.argMax(test_results, axis = 1);');
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

	// test() {
	// 	// if (this.tipo_red === 'Simple') this.createModel();
	// 	// else this.createConvModel();
	// 	//console.log(this.net);
	// 	//this.chequearEstructura();
	// 	//this.mostrar_info = !this.mostrar_info;
	// 	//this.error = false;
	// }

	entrenar() {
		if (this.chequearEstructura()) {
			this.error = false;
			this.entrenando = true;
			this.resultados = true;

			this.trainImages = this.datasetImages.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
			this.testImages = this.datasetImages.slice(this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
			this.trainLabels = this.datasetLabels.slice(0, this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
			this.testLabels = this.datasetLabels.slice(this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
			this.cargarEjemplos(0);

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
		if (this.cost_function === 'categorical_crossentropy') loss = 'categoricalCrossentropy';
		else loss = tf.losses.meanSquaredError;

		//Compilación
		this.model.compile({
			optimizer:
				this.optimizer === 'SGD' ? tf.train.sgd(this.learning_ratio) : tf.train.adam(this.learning_ratio),
			loss: loss,
			metrics: 'accuracy'
		});
		const batchSize = this.batch_size;

		// Leave out the last 15% of the training data for validation, to monitor overfitting during training.
		const validationSplit = this.ratio_val;

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
					// this.trainset_accuracy = parseFloat(logs.acc.toFixed(4)) * 100 + '%';
					this.trainset_accuracy = (logs.acc * 100).toFixed(2) + '%';
					// console.log(
					// 	`Training... (` + `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` + ` complete).`
					// );
				},
				onEpochEnd: async (epoch, logs) => {
					this.validation_accuracy = (logs.val_acc * 100).toFixed(2) + '%';
					this.trainset_accuracy = (logs.acc * 100).toFixed(2) + '%';
					// this.validation_accuracy = parseFloat(logs.val_acc.toFixed(4)) * 100 + '%';
					// this.trainset_accuracy = parseFloat(logs.acc.toFixed(4)) * 100 + '%';

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
		this.net.splice(this.net.length - 1, 0, new Layer('Dense', 1, 'ReLU', null, 0, '3x3'));
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

	actualizar_proporcion(){
		this.porc_examples_train = Math.round((1 - this.porc_examples_test)* (1-this.ratio_val) * 100);
		this.porc_examples_val = Math.round((1 - this.porc_examples_test)* this.ratio_val * 100);

		this.NUM_TEST_ELEMENTS = Math.floor(this.NUM_DATASET_ELEMENTS * this.porc_examples_test);
		this.NUM_TRAIN_ELEMENTS = this.NUM_DATASET_ELEMENTS - this.NUM_TEST_ELEMENTS;

		this.EXAMPLES_TRAIN = Math.round(this.NUM_TRAIN_ELEMENTS * (1- this.ratio_val));
		this.EXAMPLES_VAL = Math.round(this.NUM_TRAIN_ELEMENTS * this.ratio_val);
	}

	//i: entero número de la capa
	listaValidos(i) {
		let ant = this.net[i - 1].type;
		if (this.tipo_red === 'Convolucional') {
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
		} else if (this.tipo_red === 'Simple') {
			if (ant == 'Dense') return [ 'Dense', 'Dropout' ];
			else return [ 'Dense' ];
		}
	}

	chequearEstructura() {
		let salida = true;
		for (let i = 1; i < this.net.length; i++) {
			let validos = this.listaValidos(i);
			if (validos.indexOf(this.net[i]['type']) == -1) {
				this.error = true;
				this.mensaje_error = 'Error en la estructura. Conflicto entre las capas ' + (i - 1) + ' y ' + i;
				salida = false;
			}
		}
		return salida;
	}

	unsorted() {
		//Ordenador para la tabla del proceso de entrenamiento. (vacío a propósito)
		return 1;
	}
}
