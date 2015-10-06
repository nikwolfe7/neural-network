package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.NetworkElement;

public class NetworkElementLayer implements Layer {

	private NetworkElement[] elements;

	public NetworkElementLayer(NetworkElement... elements) {
		this.elements = elements;
	}

	@Override
	public void forward() {
		for(NetworkElement e : elements)
			e.forward();
	}
		
	@Override
	public void backward() {
		for(NetworkElement e : elements)
			e.backward();
	}

	@Override
	public double[] derivative() {
		double[] d = new double[elements.length];
		for(int i = 0; i < d.length; i++) {
			d[i] = elements[i].derivative();
		}
		return d;
	}

	@Override
	public double[] getOutput() {
		double[] o = new double[elements.length];
		for(int i = 0; i < o.length; i++) {
			o[i] = elements[i].getOutput();
		}
		return o;
	}

	@Override
	public double[] getErrorTerm() {
		double[] e = new double[elements.length];
		for(int i = 0; i < e.length; i++) {
			e[i] = elements[i].getErrorTerm();
		}
		return e;
	}


}
