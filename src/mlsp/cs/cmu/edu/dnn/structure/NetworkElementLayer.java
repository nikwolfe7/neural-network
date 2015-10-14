package mlsp.cs.cmu.edu.dnn.structure;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;

public class NetworkElementLayer implements Layer {

  private static final long serialVersionUID = 3296151122386671579L;
  
  private NetworkElement[] elements;

	public NetworkElementLayer(NetworkElement... elements) {
		this.elements = elements;
	}

	@Override
	public void forward() {
		for (int i = 0; i < elements.length; i++)
			elements[i].forward();
	}

	@Override
	public void backward() {
		for (int i = 0; i < elements.length; i++)
			elements[i].backward();
	}

	@Override
	public double[] derivative() {
		double[] d = new double[elements.length];
		for (int i = 0; i < d.length; i++) {
			d[i] = elements[i].derivative();
		}
		return d;
	}

	@Override
	public double[] getOutput() {
		double[] o = new double[elements.length];
		for (int i = 0; i < o.length; i++) {
			o[i] = elements[i].getOutput();
		}
		return o;
	}

	@Override
	public double[] getErrorTerm() {
		double[] e = new double[elements.length];
		for (int i = 0; i < e.length; i++) {
			e[i] = elements[i].getGradient();
		}
		return e;
	}

	@Override
	public NetworkElement[] getElements() {
		return elements;
	}

	@Override
	public int size() {
		return elements.length;
	}

	@Override
	public void addNetworkElements(NetworkElement... newElements) {
		NetworkElement[] newElementArray = new NetworkElement[elements.length + newElements.length];
		System.arraycopy(elements, 0, newElementArray, 0, elements.length);
		System.arraycopy(newElements, 0, newElementArray, elements.length, newElements.length);
		elements = newElementArray;
	}

}
