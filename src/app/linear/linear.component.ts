import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import Chart from 'chart.js';

@Component({
	selector: 'app-linear',
	templateUrl: './linear.component.html',
	styleUrls: [ './linear.component.scss' ]
})
export class LinearComponent implements OnInit {
	linearModel: tf.Sequential;
	prediction: any;

	myLineChart: Chart;

	ngOnInit(): void {
		//Called after the constructor, initializing input properties, and the first call to ngOnChanges.
		//Add 'implements OnInit' to the class.

		var ctx = document.getElementById('myChart');

		this.train();

		this.myLineChart = new Chart('myChart', {
			type: 'line',
			data: {
				labels: [],
				datasets: [
					{
						label: 'Data',
						fill: false,
						data: [],
						backgroundColor: '#168ede',
						borderColor: '#168ede'
					}
				]
			},
			options: {
				tooltips: {
					enabled: true
				},
				legend: {
					display: true,
					position: 'bottom',
					labels: {
						fontColor: 'black'
					}
				},
				scales: {
					yAxes: [
						{
							ticks: {
								fontColor: 'black'
							}
						}
					],
					xAxes: [
						{
							ticks: {
								fontColor: 'black',
								beginAtZero: true
							}
						}
					]
				}
			}
		});
	}

	async train(): Promise<any> {
		// Define a model for linear regression.
		this.linearModel = tf.sequential();
		this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [ 1 ] }));

		// Prepare the model for training: Specify the loss and the optimizer.
		this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

		// Training data, completely random stuff
		const xs = tf.tensor1d([ 3.2, 4.4, 5.5 ]);
		const ys = tf.tensor1d([ 1.6, 2.7, 3.5 ]);

		// Train
		await this.linearModel.fit(xs, ys);

		console.log('model trained!');
	}

	predict(value) {
		const val = parseInt(value);
		const output = this.linearModel.predict(tf.tensor2d([ val ], [ 1, 1 ])) as any;
		this.prediction = Array.from(output.dataSync())[0];

		this.myLineChart.data.labels.push(val);
		this.myLineChart.data.datasets[0].data.push(this.prediction);
		this.myLineChart.update();

		//console.log(this.data);
	}
}
