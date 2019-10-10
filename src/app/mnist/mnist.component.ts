import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Layer } from './../models/layer';

@Component({
	selector: 'app-mnist',
	templateUrl: './mnist.component.html',
	styleUrls: [ './mnist.component.scss' ]
})
export class MnistComponent implements OnInit {
	IMAGE_H = 28;
	IMAGE_W = 28;
	IMAGE_SIZE = 784;
	NUM_CLASSES = 10;
	NUM_DATASET_ELEMENTS = 65000;
	NUM_TRAIN_ELEMENTS = 55000;
	NUM_TEST_ELEMENTS = this.NUM_DATASET_ELEMENTS - this.NUM_TRAIN_ELEMENTS;
	datasetImages: Float32Array;
	datasetLabels: Uint8Array;
	trainImages: any;
	testImages: any;
	trainLabels: any;
	testLabels: any;
	MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
	MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

	progreso = 0;
	validation_accuracy = 0;
	test_accuracy = 0;

	model: any;
	net = [];
	arr = [];
	predictions: any;

	cantLayers = 1;

	painting = false;
	cx: CanvasRenderingContext2D;
	canvas: HTMLCanvasElement;

	constructor() {}

	ngOnInit() {
		this.net.push(new Layer('Flatten', 0, 'ReLU'));
		this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
		if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');

		console.log(this.canvas);
		console.log(this.cx);

		if (this.cx) {
			// React to mouse events on the canvas, and mouseup on the entire document
			this.canvas.addEventListener('mousedown', this.startPosition, false);
			this.canvas.addEventListener('mousemove', this.draw, false);
			this.canvas.addEventListener('mouseup', this.finishedPosition, false);
		}

		// this.canvas.addEventListener('mousedown', this.startPosition);
		// this.canvas.addEventListener('mouseup', this.finishedPosition);
		// this.canvas.addEventListener('mousemove', this.draw);
		// this.canvas.addEventListener('mousedown', function() {
		// 	this.painting = true;
		// });
		// this.canvas.addEventListener('mouseup', function() {
		// 	this.painting = false;
		// 	this.cx.beginPath();
		// });
		// this.canvas.addEventListener('mousemove', function(e) {
		// 	this.cx.lineTo(e.clientX, e.clientY);
		// 	this.cx.stroke();
		// 	this.cx.beginPath();
		// 	this.cx.moveTo(e.clientX, e.clientY);
		// });
	}

	importModel() {}

	clearCanvas() {
		if (this.cx == undefined) {
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');
		}

		this.cx.clearRect(0, 0, 400, 300);
	}

	private startPosition(e) {
		this.painting = true;
		//this.draw(e);
	}

	private finishedPosition() {
		this.painting = false;

		if (this.cx) {
			this.cx.beginPath();
		} else {
			console.log('cerrando');
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');
			this.cx.closePath();
		}
	}

	private draw(e) {
		//console.log(e);
		if (!this.painting) return;

		//console.log(this.ctx);
		// this.cx.lineWidth = 10;
		// this.cx.lineCap = 'round';
		if (this.cx) {
			//this.cx.fillStyle = 'black';
			this.cx.lineWidth = 10;
			this.cx.lineCap = 'round';
			//var w = window.innerWidth / 12;
			var w = window.scrollX + this.canvas.getBoundingClientRect().left; // X

			var z = window.scrollY + this.canvas.getBoundingClientRect().top; // Y
			this.cx.lineTo(e.clientX - w, e.clientY - z);
			this.cx.stroke();
			this.cx.beginPath();
			this.cx.moveTo(e.clientX - w, e.clientY - z);
		} else {
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');

			this.canvas.addEventListener('mousedown', this.startPosition, false);
			this.canvas.addEventListener('mousemove', this.draw, false);
			this.canvas.addEventListener('mouseup', this.finishedPosition, false);
		}
	}
	createModel() {
		const model = tf.sequential();

		model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));

		for (let i = 1; i < this.net.length; i++) {
			let layer = this.net[i];
			model.add(tf.layers.dense({ units: layer.units, activation: layer.activation }));
		}

		console.log(model);

		return model;
	}

	entrenar() {
		console.log('entrenar');
		//this.createModel();
		this.load().then(() => {
			this.model = this.createDenseModel();
			this.train(this.model, false);
		});
	}

	newLayer() {
		this.net.push(new Layer('Flatten', 0, 'ReLU'));
	}

	removeLayer(index) {
		// delete this.net[index];
		this.net.splice(index, 1);
		console.log(this.net);
	}
	onChange(value, field, index) {
		console.log(value);
		if (field == 'units') {
			this.net[index][field] = parseInt(value);
		} else {
			this.net[index][field] = value;
		}
		console.log(this.net);
	}

	createDenseModel() {
		const model = tf.sequential();
		model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));
		model.add(tf.layers.dense({ units: 42, activation: 'relu' }));
		model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
		return model;
	}

	async train(model, onIteration) {
		const optimizer = 'rmsprop';

		model.compile({
			optimizer,
			loss: 'categoricalCrossentropy',
			metrics: [ 'accuracy' ]
		});

		const batchSize = 320;

		// Leave out the last 15% of the training data for validation, to monitor
		// overfitting during training.
		const validationSplit = 0.15;

		// Get number of training epochs from the UI.
		const trainEpochs = 10;

		// We'll keep a buffer of loss and accuracy values over time.
		let trainBatchCount = 0;

		const trainData = this.getTrainData();
		const testData = this.getTestData(null);

		const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;

		let valAcc;
		await model.fit(trainData.xs, trainData.labels, {
			batchSize,
			validationSplit,
			epochs: trainEpochs,
			callbacks: {
				onBatchEnd: async (batch, logs) => {
					trainBatchCount++;
					this.progreso = Math.floor(trainBatchCount / totalNumBatches * 100);
					console.log(
						`Training... (` +
							`${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
							` complete). To stop training, refresh or close page.`
					);
					//ui.plotLoss(trainBatchCount, logs.loss, 'train');
					//ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
					if (onIteration && batch % 10 === 0) {
						onIteration('onBatchEnd', batch, logs);
					}
					await tf.nextFrame();
				},
				onEpochEnd: async (epoch, logs) => {
					valAcc = logs.val_acc;
					// ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
					// ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
					if (onIteration) {
						onIteration('onEpochEnd', epoch, logs);
					}
					await tf.nextFrame();
				}
			}
		});

		const testResult = model.evaluate(testData.xs, testData.labels);
		const testAccPercent = testResult[1].dataSync()[0] * 100;
		const finalValAccPercent = valAcc * 100;
		this.validation_accuracy = parseFloat(finalValAccPercent.toFixed(2));
		this.test_accuracy = parseFloat(testAccPercent.toFixed(2));
		console.log(
			`Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
				`Final test accuracy: ${testAccPercent.toFixed(1)}%` +
				`Test result: ${testResult}`
		);
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
			xs = xs.slice([ 0, 0, 0, 0 ], [ numExamples, this.IMAGE_H, this.IMAGE_W, 1 ]);
			labels = labels.slice([ 0, 0 ], [ numExamples, this.NUM_CLASSES ]);
		}
		return { xs, labels };
	}

	mostrar() {
		if (!this.cx) {
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');
		}

		var imageData = this.cx.getImageData(0, 0, this.canvas.width, this.canvas.height);
		console.log(imageData);
	}

	async predict() {
		if (!this.cx) {
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			if (this.canvas.getContext) this.cx = this.canvas.getContext('2d');
		}

		const pred = await tf.tidy(() => {
			let imageData = this.cx.getImageData(0, 0, this.canvas.width, this.canvas.height);
			let img = tf.browser.fromPixels(imageData, 1).expandDims(0);
			//let imgR = tf.reshape(img, [ 28, 28 ]);
			let imgR = tf.cast(img, 'float32');
			console.log(imgR);
			console.log(this.model);

			const output = this.model.predict(imgR) as any;

			this.predictions = Array.from(output.dataSync());
			console.log(this.predictions);
			console.log(output);
		});
	}

	// protected async predict(imageData: ImageData) {

	// 	const pred = await tf.tidy(() => {

	// 	  let img:any = tf.fromPixels(imageData, 1);
	// 	  img = img.reshape([1, 28, 28, 1]);
	// 	  img = tf.cast(img, 'float32');

	// 	  const output = this.model.predict(img) as any;

	// 	  this.predictions = Array.from(output.dataSync());
	// 	});
	// }
}