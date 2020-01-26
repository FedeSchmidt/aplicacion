export class Layer {
	// type: tipo de layer
	// units: cantidad de neuronas de la capa
	// activation: función de activación
	// input_shape: solo para flatten
	// ratio: solo para dropout, ratio de desconexion
	constructor(
		public type: string,
		public units: number,
		public activation: string,
		public input_shape,
		public ratio: number
	) {}
}
