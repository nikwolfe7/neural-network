package mlsp.cs.cmu.edu.dnn.util;

import java.util.Arrays;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SimpleEdge;
import mlsp.cs.cmu.edu.dnn.factory.NetworkElementAbstractFactory;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NetworkElementLayer;

public class LayerElementUtils {

	/**
	 * Returns the weight matrix between two layers after fully connecting them
	 * Matrix is returned as a 1D array which can be indexed as a 2D array as
	 * follows:
	 * 
	 * row = neuron in the layer, i.e. "to" col = element in the previous layer,
	 * i.e. "from"
	 * 
	 * Edge get(int row, int col, int numcols) { return weightMatrix[row *
	 * numcols + col]; }
	 * 
	 * @param fromLayer
	 * @param toLayer
	 * @param factory
	 * @return
	 */
	public static Layer connect(Layer fromLayer, Layer toLayer, NetworkElementAbstractFactory factory) {
		int rows, cols;
		rows = toLayer.size();
		cols = fromLayer.size();
		Edge[] weightMatrix = new Edge[rows * cols];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				Neuron out = (Neuron) toLayer.getElements()[row];
				Neuron in = (Neuron) fromLayer.getElements()[col];
				Edge w = factory.getNewEdge();
				attachElements(in, w, out);
				weightMatrix[row * cols + col] = w;
			}
		}
		return new NetworkElementLayer(weightMatrix);
	}

	/**
	 * Returns the weight matrix between two layers after fully connecting them
	 * Matrix is returned as a 1D array which can be indexed as a 2D array as
	 * follows:
	 * 
	 * row = neuron in the layer, i.e. "to" col = element in the previous layer,
	 * i.e. "from"
	 * 
	 * Edge get(int row, int col, int numcols) { return weightMatrix[row *
	 * numcols + col]; }
	 * 
	 * @param fromLayer
	 * @param toLayer
	 * @return
	 */
	public static Layer connect(Layer fromLayer, Layer toLayer) {
		int rows, cols;
		rows = toLayer.size();
		cols = fromLayer.size();
		Edge[] weightMatrix = new Edge[rows * cols];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				Neuron out = (Neuron) toLayer.getElements()[row];
				Neuron in = (Neuron) fromLayer.getElements()[col];
				Edge w = new SimpleEdge();
				attachElements(in, w, out);
				weightMatrix[row * cols + col] = w;
			}
		}
		return new NetworkElementLayer(weightMatrix);
	}
	
	/**
	 * Converts an edge from one type to another given two instantiated
	 * Edges of potentially different subclasses
	 * 
	 * @param from
	 * @param to
	 * @return
	 */
	public static void convertEdge(Edge from, Edge to) {
	  to.setBatchUpdate(from.isBatchUpdate());
	  to.setGradient(from.getGradient());
	  to.setLearningRate(from.getLearningRate());
	  to.setWeight(from.getWeight());
	  Neuron fN, tN;
	  fN = (Neuron) from.getIncomingElement();
	  tN = (Neuron) from.getOutgoingElement();
	  fN.removeElement(from);
	  tN.removeElement(from);
	  attachElements(fN, to, tN);
	}
	
	public static void convertNeuron(Neuron from, Neuron to) {
    for(NetworkElement e : from.getIncomingElements()) {
      Edge edge = (Edge) e;
      edge.setOutgoingElement(to);
      to.addIncomingElement(edge);
    }
    for(NetworkElement e : from.getOutgoingElements()) {
      Edge edge = (Edge) e;
      edge.setIncomingElement(to);
      to.addOutgoingElement(edge);
    }
    to.setOutput(from.getOutput());
    to.setGradient(from.getGradient());
	}
	
	/**
	 * 
	 * @param o1
	 * @param o2
	 */
	public static void convertOutput(Output from, Output to) {
	  convertNeuron(from, to);
	  to.setTruthValue(from.getTruthValue());
  }

	/**
	 * Attach two individual neurons together with an Edge
	 * 
	 * @param in
	 * @param w
	 * @param out
	 */
	public static void attachElements(Neuron in, Edge w, Neuron out) {
		w.setIncomingElement(in);
		w.setOutgoingElement(out);
		in.addOutgoingElement(w);
		out.addIncomingElement(w);
	}

	public static NetworkElement[] removeLayerElement(Layer networkElementLayer, NetworkElement e) {
		List<NetworkElement> elements = Arrays.asList(networkElementLayer.getElements());
		if (!elements.contains(e)) {
			return networkElementLayer.getElements(); /* unchanged */
		} else {
			int i = 0, size = Math.max(0, elements.size() - 1);
			NetworkElement[] newElements = new NetworkElement[size];
			for (NetworkElement element : elements) {
				if (element != e) {
					newElements[i++] = element;
				} else {
					if (e instanceof Neuron) {
						List<NetworkElement> incoming = ((Neuron) e).getIncomingElements();
						List<NetworkElement> outgoing = ((Neuron) e).getOutgoingElements();
						for (NetworkElement elem : incoming) {
							Edge edge = (Edge) elem;
							Neuron n = (Neuron) edge.getIncomingElement();
							n.getOutgoingElements().remove(edge);
							edge.setIncomingElement(null);
							edge.setOutgoingElement(null);
						}
						for (NetworkElement elem : outgoing) {
							Edge edge = (Edge) elem;
							Neuron n = (Neuron) edge.getOutgoingElement();
							n.getIncomingElements().remove(edge);
							edge.setIncomingElement(null);
							edge.setOutgoingElement(null);
						}
					} else if (e instanceof Edge) {
						Neuron next = (Neuron) ((Edge) e).getOutgoingElement();
						Neuron prev = (Neuron) ((Edge) e).getIncomingElement();
						if (prev != null && next != null) {
							next.getIncomingElements().remove(e);
							prev.getOutgoingElements().remove(e);
						}
					}
				}
			}
			return newElements;
		}
	}
}
