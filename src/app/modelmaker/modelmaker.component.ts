import { Component, OnInit } from '@angular/core';
import { Layer } from './../models/layer';
import * as tf from '@tensorflow/tfjs';

@Component({
	selector: 'app-modelmaker',
	templateUrl: './modelmaker.component.html',
	styleUrls: [ './modelmaker.component.scss' ]
})
export class ModelmakerComponent implements OnInit {
	// parámetros generales
	learning_ratio = 0.01;
	epochs = 10;
	metrics = 'accuracy';
	cost_function = 'Cross Entropy';
	batch_check = true;
	batch_size = 10;

	// modelo
	model: any;
	net = [];
	cantLayers = 1;

	constructor() {}

	ngOnInit() {
		// this.net.push(new Layer('Flatten', 0, 'ReLU'));
	}

	newLayer() {
		// this.net.push(new Layer('Flatten', 0, 'ReLU'));
	}

	removeLayer(index) {
		this.net.splice(index, 1);
	}
	onChange(value, field, index) {
		if (field == 'units') {
			this.net[index][field] = parseInt(value);
		} else {
			this.net[index][field] = value;
		}
	}

	// //de la red.
	// createDenseModel() {
	// 	const model = tf.sequential();
	// 	model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));
	// 	model.add(tf.layers.dense({ units: 42, activation: 'relu' }));
	// 	model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
	// 	return model;
	// }

	// createModel() {
	// 	const model = tf.sequential();

	// 	model.add(tf.layers.flatten({ inputShape: [ this.IMAGE_H, this.IMAGE_W, 1 ] }));

	// 	for (let i = 1; i < this.net.length; i++) {
	// 		let layer = this.net[i];
	// 		model.add(tf.layers.dense({ units: layer.units, activation: layer.activation }));
	// 	}

	// 	console.log(model);

	// 	return model;
	// }

	// entrenar() {
	// 	console.log('entrenar');
	// 	//this.createModel();
	// 	this.load().then(() => {
	// 		//const trainData = this.getTrainData();
	// 		//console.log(trainData.xs);
	// 		//const tensorData = trainData.xs.dataSync();

	// 		//console.log(tensorData);
	// 		this.model = this.createDenseModel();
	// 		this.train(this.model, false);
	// 	});
	// }

	// async train(model, onIteration) {
	// 	//const optimizer = 'rmsprop';

	// 	//optimizer: tf.train.adam(0.001),
	// 	model.compile({
	// 		optimizer: tf.train.sgd(0.15),
	// 		loss: 'categoricalCrossentropy',
	// 		metrics: [ 'accuracy' ]
	// 	});

	// 	const batchSize = 64;

	// 	// Leave out the last 15% of the training data for validation, to monitor
	// 	// overfitting during training.
	// 	const validationSplit = 0.15;

	// 	// Get number of training epochs from the UI.
	// 	const trainEpochs = 30;

	// 	// We'll keep a buffer of loss and accuracy values over time.
	// 	let trainBatchCount = 0;

	// 	const trainData = this.getTrainData();
	// 	const testData = this.getTestData(null);

	// 	const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;

	// 	let valAcc;
	// 	console.log(trainData.xs);
	// 	console.log(trainData.labels);
	// 	await model.fit(trainData.xs, trainData.labels, {
	// 		batchSize,
	// 		validationSplit,
	// 		epochs: trainEpochs,
	// 		callbacks: {
	// 			onBatchEnd: async (batch, logs) => {
	// 				trainBatchCount++;
	// 				this.progreso = Math.floor(trainBatchCount / totalNumBatches * 100);
	// 				console.log(
	// 					`Training... (` +
	// 						`${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
	// 						` complete). To stop training, refresh or close page.`
	// 				);
	// 				//ui.plotLoss(trainBatchCount, logs.loss, 'train');
	// 				//ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
	// 				if (onIteration && batch % 10 === 0) {
	// 					onIteration('onBatchEnd', batch, logs);
	// 				}
	// 				await tf.nextFrame();
	// 			},
	// 			onEpochEnd: async (epoch, logs) => {
	// 				valAcc = logs.val_acc;
	// 				// ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
	// 				// ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
	// 				if (onIteration) {
	// 					onIteration('onEpochEnd', epoch, logs);
	// 				}
	// 				await tf.nextFrame();
	// 			}
	// 		}
	// 	});

	// 	const testResult = model.evaluate(testData.xs, testData.labels);
	// 	const testAccPercent = testResult[1].dataSync()[0] * 100;
	// 	const finalValAccPercent = valAcc * 100;
	// 	this.validation_accuracy = parseFloat(finalValAccPercent.toFixed(2));
	// 	this.test_accuracy = parseFloat(testAccPercent.toFixed(2));
	// 	console.log(
	// 		`Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
	// 			`Final test accuracy: ${testAccPercent.toFixed(1)}%` +
	// 			`Test result: ${testResult}`
	// 	);
	// }

	// async load() {
	// 	const img = new Image();
	// 	const canvas = document.createElement('canvas');
	// 	const ctx = canvas.getContext('2d');

	// 	const imgRequest = new Promise((resolve, reject) => {
	// 		img.crossOrigin = '';
	// 		img.onload = () => {
	// 			img.width = img.naturalWidth;
	// 			img.height = img.naturalHeight;

	// 			const datasetBytesBuffer = new ArrayBuffer(this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE * 4);

	// 			const chunkSize = 5000;
	// 			canvas.width = img.width;
	// 			canvas.height = chunkSize;

	// 			for (let i = 0; i < this.NUM_DATASET_ELEMENTS / chunkSize; i++) {
	// 				const datasetBytesView = new Float32Array(
	// 					datasetBytesBuffer,
	// 					i * this.IMAGE_SIZE * chunkSize * 4,
	// 					this.IMAGE_SIZE * chunkSize
	// 				);
	// 				ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

	// 				const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

	// 				for (let j = 0; j < imageData.data.length / 4; j++) {
	// 					// All channels hold an equal value since the image is grayscale, so
	// 					// just read the red channel.
	// 					datasetBytesView[j] = imageData.data[j * 4] / 255;
	// 				}
	// 			}
	// 			this.datasetImages = new Float32Array(datasetBytesBuffer);

	// 			resolve();
	// 		};
	// 		img.src = this.MNIST_IMAGES_SPRITE_PATH;
	// 	});

	// 	//console.log(this.datasetImages);

	// 	const labelsRequest = fetch(this.MNIST_LABELS_PATH);
	// 	const [ imgResponse, labelsResponse ] = await Promise.all([ imgRequest, labelsRequest ]);

	// 	this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

	// 	// Slice the the images and labels into train and test sets.
	// 	this.trainImages = this.datasetImages.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);

	// 	//console.log(this.trainImages);
	// 	this.testImages = this.datasetImages.slice(this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
	// 	this.trainLabels = this.datasetLabels.slice(0, this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
	// 	this.testLabels = this.datasetLabels.slice(this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
	// }

	// getTrainData() {
	// 	const xs = tf.tensor4d(this.trainImages, [
	// 		this.trainImages.length / this.IMAGE_SIZE,
	// 		this.IMAGE_H,
	// 		this.IMAGE_W,
	// 		1
	// 	]);
	// 	const labels = tf.tensor2d(this.trainLabels, [ this.trainLabels.length / this.NUM_CLASSES, this.NUM_CLASSES ]);
	// 	console.log(xs);
	// 	console.log(labels);
	// 	return { xs, labels };
	// }

	// getTestData(numExamples) {
	// 	let xs = tf.tensor4d(this.testImages, [
	// 		this.testImages.length / this.IMAGE_SIZE,
	// 		this.IMAGE_H,
	// 		this.IMAGE_W,
	// 		1
	// 	]);
	// 	let labels = tf.tensor2d(this.testLabels, [ this.testLabels.length / this.NUM_CLASSES, this.NUM_CLASSES ]);

	// 	if (numExamples != null) {
	// 		xs = xs.slice([ 0, 0, 0, 0 ], [ numExamples, this.IMAGE_H, this.IMAGE_W, 1 ]);
	// 		labels = labels.slice([ 0, 0 ], [ numExamples, this.NUM_CLASSES ]);
	// 	}
	// 	return { xs, labels };
	// }
}
