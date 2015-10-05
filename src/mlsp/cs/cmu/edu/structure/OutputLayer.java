package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Output;

public class OutputLayer extends Layer {

  private Output[] outputs;
  
  public OutputLayer(Output... elements) {
    super(elements);
    this.outputs = elements;
  }
  
  public void setTruthValue(double[] truthVector) {
    for(int i = 0; i < truthVector.length; i++) {
      outputs[i].setTruthValue(truthVector[i]);
    }
  }

}
