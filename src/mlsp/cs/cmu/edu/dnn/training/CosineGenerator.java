package mlsp.cs.cmu.edu.dnn.training;

import java.util.Random;

public class CosineGenerator implements DataInstanceGenerator {

  Random rnd = new Random();
  
  @Override
  public DataInstance getNewDataInstance() {
    double val = rnd.nextDouble();
    double cos = Math.cos(val);
    double[] x = new double[] {cos, val};
    return new DataInstance(1, 1, x);
  }

}
