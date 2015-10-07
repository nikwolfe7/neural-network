package mlsp.cs.cmu.edu.elements;

public class ThresholdOutput extends LinearOutput {
  
  private double threshold = 0.75;
  
  @Override
  public void forward() {
    super.forward();
    double o = getOutput();
    if(o > (threshold))
      setOutput(0.99);
    else if (o < (1-threshold))
      setOutput(0.1);
  }

}
