package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.*;

public interface NetworkElementAbstractFactory {
  
  public Input getNewInput();
  
  public Output getNewOutput();
  
  public Neuron getNewNeuron();
  
  public Edge getNewEdge();
  
  public Bias getNewBias();

}
