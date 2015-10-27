package mlsp.cs.cmu.edu.dnn.structure;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;

public class CompositeLayer implements Layer {

	private static final long serialVersionUID = 6555068509293432784L;

	private List<Layer> subLayers;
	private int totalSize;

	public CompositeLayer(Layer... layers) {
		this.totalSize = 0;
		this.subLayers = new ArrayList<Layer>();
		for (Layer l : layers) {
			subLayers.add(l);
			totalSize += l.size();
		}
	}

	public void addLayer(Layer layer) {
		subLayers.add(layer);
		totalSize += layer.size();
	}

	@Override
	public void forward() {
		for (Layer l : subLayers)
			l.forward();
	}

	@Override
	public void backward() {
		for (Layer l : subLayers)
			l.backward();
	}

	@Override
	public double[] derivative() {
		double[] derivatives = new double[totalSize];
		int start = 0;
		for (Layer l : subLayers) {
			double[] d = l.derivative();
			System.arraycopy(d, 0, derivatives, start, d.length);
			start += d.length;
		}
		return derivatives;
	}

	@Override
	public double[] getOutput() {
		double[] outputs = new double[totalSize];
		int start = 0;
		for (Layer l : subLayers) {
			double[] o = l.getOutput();
			System.arraycopy(o, 0, outputs, start, o.length);
			start += o.length;
		}
		return outputs;
	}

	@Override
	public double[] getGradient() {
		double[] gradients = new double[totalSize];
		int start = 0;
		for (Layer l : subLayers) {
			double[] g = l.getGradient();
			System.arraycopy(g, 0, gradients, start, g.length);
			start += g.length;
		}
		return gradients;
	}

	public List<Layer> getSubLayers() {
		return subLayers;
	}

	@Override
	public NetworkElement[] getElements() {
		NetworkElement[] layerElements = new NetworkElement[totalSize];
		int start = 0;
		for (Layer l : subLayers) {
			NetworkElement[] e = l.getElements();
			System.arraycopy(e, 0, layerElements, start, e.length);
			start += e.length;
		}
		return layerElements;
	}

	/**
	 * This will have the same effect as {@link #addLayer(Layer)}
	 */
	@Override
	public void addNetworkElements(NetworkElement... elements) {
		Layer layer = new NetworkElementLayer(elements);
		addLayer(layer);
	}

	@Override
	public int size() {
		return totalSize;
	}

  @Override
  public void removeNetworkElement(NetworkElement e) {
    for(Layer layer : subLayers) {
      layer.removeNetworkElement(e);
    }
  }

}
