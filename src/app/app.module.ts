import { InformacionComponent } from './informacion/informacion.component';
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MnistComponent } from './mnist/mnist.component';
import { NavbarComponent } from './navbar/navbar.component';

const appRoutes: Routes = [
	{ path: '', component: MnistComponent },
	{ path: 'info', component: InformacionComponent }
];

@NgModule({
	declarations: [ AppComponent, MnistComponent, NavbarComponent, InformacionComponent ],
	imports: [
		BrowserModule,
		BrowserAnimationsModule,
		RouterModule.forRoot(appRoutes),
		MatSliderModule,
		MatButtonModule,
		MatInputModule,
		FormsModule
	],
	providers: [],
	bootstrap: [ AppComponent ]
})
export class AppModule {}
