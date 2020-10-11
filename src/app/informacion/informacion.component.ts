import { Component, OnInit } from '@angular/core';
import Datos from '../../assets/info.json';
@Component({
	selector: 'app-informacion',
	templateUrl: './informacion.component.html',
	styleUrls: [ './informacion.component.scss' ]
})
export class InformacionComponent implements OnInit {
	expanded = {
		mnist: true,
		learning_rate: false,
		metric: false,
		ciclos: false,
		cost_function: false,
		batch_size: false,
		estructura: false,
		activacion: false,
		construccion: false,
		tipos: false,
		optimizador: false,
		overfitting: false
	};

	data;

	constructor() {}

	ngOnInit() {
		let d = JSON.stringify(Datos);
		this.data = JSON.parse(d)[0];
	}

	show(source) {
		this.expanded[source] = !this.expanded[source];
	}
}
