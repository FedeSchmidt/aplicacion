import { Component, OnInit } from '@angular/core';
import { Layer } from './../models/layer';

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
		this.net.push(new Layer('Flatten', 0, 'ReLU'));
	}

	newLayer() {
		this.net.push(new Layer('Flatten', 0, 'ReLU'));
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
}
