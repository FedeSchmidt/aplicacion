import { Component, OnInit } from '@angular/core';
import { CdkTextareaAutosize } from '@angular/cdk/text-field';

@Component({
	selector: 'app-codigo',
	templateUrl: './codigo.component.html',
	styleUrls: [ './codigo.component.scss' ]
})
export class CodigoComponent implements OnInit {
	painting = false;
	canvas;
	cx;
	constructor() {}

	ngOnInit() {
		this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
		this.cx = this.canvas.getContext('2d');

		this.canvas.height = 200;
		this.canvas.width = 200;

		this.canvas.addEventListener('mousedown', this.startPosition);
		this.canvas.addEventListener('mouseup', this.finishedPosition);
		this.canvas.addEventListener('mousemove', this.draw);
	}

	startPosition() {
		this.painting = true;
	}

	finishedPosition() {
		this.painting = false;

		this.cx.beginPath();
	}

	draw(e) {
		if (!this.painting) return;
		if (this.canvas == undefined) {
			this.canvas = <HTMLCanvasElement>document.getElementById('canvas');
			this.cx = this.canvas.getContext('2d');
		}

		this.cx.lineWidth = 8;
		this.cx.lineCap = 'round';
		this.cx.lineTo(e.clientX - this.canvas.offsetLeft, e.clientY - this.canvas.offsetTop);
		this.cx.stroke();
		this.cx.beginPath();
		this.cx.moveTo(e.clientX - this.canvas.offsetLeft, e.clientY - this.canvas.offsetTop);
	}
}
