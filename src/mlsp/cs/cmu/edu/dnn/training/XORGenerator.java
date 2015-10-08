package mlsp.cs.cmu.edu.dnn.training;

import java.util.Random;

public class XORGenerator implements DataInstanceGenerator {

  private Random rnd = new Random();
  private double[][] truthTable = new double[][] {{0, 1},{1, 0}};
  
  @Override
  public DataInstance getNewDataInstance() {
    double x1, x2, y;
    x1 = rnd.nextInt(2);
    x2 = rnd.nextInt(2);
    y = truthTable[(int) x1][(int) x2];
    return new DataInstance(2, 1, new double[] {y, x1, x2});
  }

}
