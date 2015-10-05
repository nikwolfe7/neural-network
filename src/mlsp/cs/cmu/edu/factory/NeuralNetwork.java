package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Output;

public class NeuralNetwork {
  
  private List<NetworkElement[]> network;
  private Input[] inputLayer;
  private Output[] outputLayer;
  
  public NeuralNetwork(NetworkElement[]... layers) {
	/* using ArrayList b/c the width varies */
    this.network = new ArrayList<NetworkElement[]>();
    for(int i = 0; i < layers.length; i++) {
      network.add(layers[i]); 
      if(i == 0) {
        this.inputLayer = (Input[]) layers[i];
      } else if (i == layers.length - 1) {
        this.outputLayer = (Output[]) layers[i];
      } 
    }
  }
  
  


}
