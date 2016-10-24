package mlsp.cs.cmu.edu.dnn.structure;

import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.GainSwitchNeuron;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SecondDerivativeOutput;
import mlsp.cs.cmu.edu.dnn.elements.SwitchEdge;
import mlsp.cs.cmu.edu.dnn.elements.Switchable;

public class GainSwitchLayer implements Layer {

  private static final long serialVersionUID = -6176626511061680845L;

  private Layer layer;
  
  public GainSwitchLayer(Layer layer) {
    NetworkElement[] elements = layer.getElements();
    NetworkElement[] newElements = new NetworkElement[elements.length];
    for(int i = 0; i < elements.length; ++i) {
      NetworkElement e = elements[i];
      if(e instanceof Neuron) {
        if(!(e instanceof Output) && !(e instanceof Input) && !(e instanceof Bias)) {
          GainSwitchNeuron neuron = new GainSwitchNeuron((Neuron) e);
          newElements[i] = neuron;
        } else if(e instanceof Output) {
          SecondDerivativeOutput output = new SecondDerivativeOutput((Output) e);
          newElements[i] = output;
        } else {
          newElements[i] = e;
        }
      } else if(e instanceof Edge) {
        SwitchEdge edge  = new SwitchEdge((Edge) e);
        newElements[i] = edge;
      } else {
        newElements[i] = e;
      }
    }
    this.layer = new NetworkElementLayer(newElements);
  }
  
  public void resetLayer() {
    for(NetworkElement e : getElements()) {
      if(e instanceof GainSwitchNeuron) {
        ((GainSwitchNeuron) e).reset();
      }
    }
  }
  
  public void switchOff(boolean b) {
    for(NetworkElement e : getElements()) {
      if(e instanceof Switchable) {
        ((Switchable) e).setSwitchedOff(b);
      }
    }
  }

  @Override
  public void forward() {
    layer.forward();
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
