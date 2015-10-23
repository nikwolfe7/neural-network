package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.CostFunction;

public class CrossEntropyOutput extends LinearOutput {

  private static final long serialVersionUID = 7403305380238578951L;
  
  @Override
  public double derivative() {
    return CostFunction.crossEntropyDerivative(getOutput(), getTruthValue());
  }

}
