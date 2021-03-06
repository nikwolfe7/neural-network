package mlsp.cs.cmu.edu.dnn.structure;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.Output;

public class SoftMaxLayer implements Layer {

	private static final long serialVersionUID = -4047494397180563664L;
	
	private Layer layer;
	
	public SoftMaxLayer(Layer layer) {
		this.layer = layer;
	}

	@Override
	public void forward() {
		layer.forward();
		double sum = 0;
		for(double d : getOutput())
			sum += Math.exp(d);
		sum = (sum > 0) ? sum : Double.MIN_NORMAL;
		for(NetworkElement e : getElements()) {
			Output output;
			output = (Output) e;
			double o = output.getOutput();
			double softMax = Math.exp(o) / sum;
			output.setOutput(softMax);
		}
	}

	@Override
	public void backward() {
		layer.backward();
	}

	@Override
	public double[] derivative() {
		return layer.derivative();
	}

	@Override
	public double[] getOutput() {
		return layer.getOutput();
	}

	@Override
	public double[] getGradient() {
		return layer.getGradient();
	}

	@Override
	public NetworkElement[] getElements() {
		return layer.getElements();
	}

	@Override
	public void addNetworkElements(NetworkElement... elements) {
		layer.addNetworkElements(elements);
	}

	@Override
	public int size() {
		return layer.size();
	}

  @Override
  public void removeNetworkElement(NetworkElement e) {
    layer.removeNetworkElement(e);
  }

}
