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
    this.network = new ArrayList<NetworkElement[]>();
    for(int i = 0; i < layers.length; i++) {
      if(i == 0) {
        
      } else if (i == layers.length - 1) {
        
      } else {
        
      }
    }
  }


}
