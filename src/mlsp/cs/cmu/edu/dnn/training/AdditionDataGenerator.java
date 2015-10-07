package mlsp.cs.cmu.edu.dnn.training;

import java.util.Random;

public class AdditionDataGenerator implements DataInstanceGenerator {

  private Random rnd;
  private int numVals;
  private int bound = 10;
  
  public AdditionDataGenerator(int numValues) {
    this.rnd = new Random();
    this.numVals = numValues;
  }

  @Override
  public DataInstance getNewDataInstance() {
    double[] arr = new double[numVals + 1];
    for(int i = 1; i < arr.length; i++) {
      double val = rnd.nextInt(bound);
      arr[i] = val;
      arr[0] += val;
    }
    return new DataInstance(numVals, 1, arr);
  }

}
