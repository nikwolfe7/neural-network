package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;

public abstract class Layer {

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
  
  public Edge[][] connect(Layer layer) {
    Neuron[] nextLayer = layer.getLayerElements();
    Edge[][] weightMatrix = new Edge[elements.length][nextLayer.length];
    for (int i = 0; i < elements.length; i++) {
      for (int j = 0; j < nextLayer.length; j++) {
        Edge e = new Edge();
        connectElements(elements[i], e, nextLayer[j]);
        weightMatrix[i][j] = e;
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
