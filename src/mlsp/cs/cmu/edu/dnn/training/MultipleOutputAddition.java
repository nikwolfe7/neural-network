package mlsp.cs.cmu.edu.dnn.training;

import java.util.Random;

public class MultipleOutputAddition implements DataInstanceGenerator {

  private Random rnd = new Random();
  private int bound = 10;
  
  @Override
  public DataInstance getNewDataInstance() {
    double[] arr = new double[6];
    double x1, x2;
    double x3, x4;
    double y1, y2;
    x1 = rnd.nextInt(bound);
    x2 = rnd.nextInt(bound);
    x3 = rnd.nextInt(bound);
    x4 = rnd.nextInt(bound);
    y1 = x1 + x2;
    y2 = x3 + x4;
    arr = new double[] {y1, y2, x1, x2, x3, x4};
    return new DataInstance(4, 2, arr);
  }

}
