package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.CrossEntropySoftmaxOutput;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;

public class CrossEntropyNetworkFactory implements NetworkElementAbstractFactory {

  @Override
  public Input getNewInput() {
    return new Input();
  }

  @Override
  public Output getNewOutput() {
    return new CrossEntropySoftmaxOutput();
  }

  @Override
  public Neuron getNewNeuron() {
    return new Neuron();
  }

  @Override
  public Edge getNewEdge() {
    return new Edge();
  }

  @Override
  public Bias getNewBias() {
    return new Bias();
  }

}
