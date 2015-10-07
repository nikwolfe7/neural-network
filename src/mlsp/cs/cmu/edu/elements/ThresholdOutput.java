package mlsp.cs.cmu.edu.elements;

public class ThresholdOutput extends LinearOutput {
  
  private double threshold = 0.55;
  
  @Override
  public void forward() {
    super.forward();
//    double o = getOutput();
//    if(o > (threshold))
//      setOutput(0.99);
//    else
//      setOutput(0.01);
  }

}
