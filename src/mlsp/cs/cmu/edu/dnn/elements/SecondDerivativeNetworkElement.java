package mlsp.cs.cmu.edu.dnn.elements;

public interface SecondDerivativeNetworkElement extends NetworkElement {
  
  /**/
  public double secondDerivative();
  
  /**/
  public double getSecondGradient();
  
}
