import { Component, OnInit } from '@angular/core';

@Component({
	selector: 'app-codigo',
	templateUrl: './codigo.component.html',
	styleUrls: [ './codigo.component.scss' ]
})
export class CodigoComponent implements OnInit {
	y = 1;

	x = 1;

	t;

	constructor() {}

	ngOnInit() {
		console.log(this.x);
		console.log(this.test);
		this.test(1);
		console.log(this.x);

		this.t = String(this.test).replace(/param/g, this.y + '');
		//this.t = this.test.toString().replace(/param/g, this.y + '');
		console.log(this.t);
	}

	test(param) {
		this.x = this.y + 1;
		console.log(param);
	}
}
