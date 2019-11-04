import { Component, OnInit, Input } from '@angular/core';

@Component({
	selector: 'app-ejemplos',
	templateUrl: './ejemplos.component.html',
	styleUrls: [ './ejemplos.component.scss' ]
})
export class EjemplosComponent implements OnInit {
	@Input() labels;
	@Input() nuevas;
	@Input() examples_length;
	@Input() data;

	examples_test = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ];

	constructor() {}

	ngOnInit() {
		console.log(this.labels);
		console.log(this.nuevas);
		console.log(this.examples_length);
		console.log(this.data);
		for (let i = 0; i < this.examples_length; i++) {
			const image = this.data.xs.slice([ i, 0 ], [ 1, this.data.xs.shape[1] ]);

			this.draw(image.flatten(), i);
		}
	}

	draw(image, i) {
		let canvas = <HTMLCanvasElement>document.getElementById('canvas' + i);

		console.log(canvas);
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
}
