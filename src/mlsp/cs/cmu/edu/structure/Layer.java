package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;

public class Layer {

  private Neuron[] elements;
  
  public Layer(Neuron... elements) {
    this.elements = elements;
  }
  
  public void forwardPropagate() {
    for(int i = 0; i < elements.length; i++) {
      elements[i].forward();
    }
  }
  
  public void backPropagate() {
    for(int i = 0; i < elements.length; i++) {
      elements[i].backward();
    }
  }
  
  public double[] getOutput() {
    double[] output = new double[elements.length];
    for(int i = 0; i < elements.length; i++) {
      output[i] = elements[i].getOutput();
    }
    return output;
  }
  
  public Neuron[] getLayerElements() {
    return elements;
  }
  
  public Edge[] connect(Layer layer) {
    Neuron[] nextLayer = layer.getLayerElements();
    Edge[] weightMatrix = new Edge[elements.length * nextLayer.length];
    int ncols = nextLayer.length;
    for (int row = 0; row < elements.length; row++) {
      for (int col = 0; col < nextLayer.length; col++) {
        Edge e = new Edge();
        connectElements(elements[row], e, nextLayer[col]);
        weightMatrix[row * ncols + col] = e;
      }
    }
    return weightMatrix;
  }
  
  private void connectElements(Neuron in, Edge w, Neuron out) {
    w.setIncomingElement(in);
    w.setOutgoingElement(out);
    in.addOutgoingElement(w);
    out.addIncomingElement(w);
  }

}
