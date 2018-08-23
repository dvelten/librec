/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.recommender.cf.ranking;


import com.google.common.collect.BiMap;
import net.librec.common.LibrecException;
import net.librec.math.structure.SparseVector;
import net.librec.recommender.AbstractRecommender;
import net.librec.recommender.item.ItemEntry;
import net.librec.recommender.item.RecommendedItemList;
import net.librec.recommender.item.RecommendedList;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;


/**
 * <p>
 *     Nonnegative Matrix Factorization of the item to item purchase matrix.
 * </p>
 *
 * <p>
 * (only implicit or binary input supported)
 * </p>
 *
 * <p>
 * NMFItemItem uses as model of the probability distribution P(V) ~ W * H * V
 * </p>
 *
 * <p>
 * Where V is the observed purchase user item matrix.
 * And W and H are trained matrices.
 * </p>
 *
 * <p>
 * H is the matrix for 'analyzing' the current purchase item history and calculates the assumed latent feature vector.<br>
 * W is the matrix for 'estimating' the purchase probability of the next item calculated from the latent feature vector.
 * </p>
 *
 * <p>
 * In contrast to this the original Nonnegative Matrix Factorization is a factorization of the item - user matrix.
 * </p>
 *
 * <p>Literature:</p>
 * <ul>
 * <li>Lee, Daniel D., and H. Sebastian Seung. "Learning the parts of objects by non-negative matrix factorization." Nature 401.6755 (1999): 788.</li>
 * <li>Yuan, Zhijian, and Erkki Oja. "Projective nonnegative matrix factorization for image compression and feature extraction." Image analysis (2005): 333-342.</li>
 * <li>Yang, Zhirong, Zhijian Yuan, and Jorma Laaksonen. "Projective non-negative matrix factorization with applications to facial image processing." International Journal of Pattern Recognition and Artificial Intelligence 21.08 (2007): 1353-1362.</li>
 * <li>Yang, Zhirong, and Erkki Oja. "Unified development of multiplicative algorithms for linear and quadratic nonnegative matrix factorization." IEEE transactions on neural networks 22.12 (2011): 1878-1891.</li>
 * <li>Zhang, He, Zhirong Yang, and Erkki Oja. "Adaptive multiplicative updates for projective nonnegative matrix factorization." International Conference on Neural Information Processing. Springer, Berlin, Heidelberg, 2012.</li>
 * <li>Kabbur, Santosh, Xia Ning, and George Karypis. "FISM: Factored Item Similarity Models for top-n recommender systems." Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013.</li>
 * </ul>
 *
 *
 * <p>
 * Item-Item models are much better usable for fast online recommendation with a lot of new and/or fast changing users.<br>
 * (Solves the cold start problem of new users)
 * </p>
 *
 * <p>
 * There are some optimisation switches:
 * </p>
 *
 * <p>
 * rec.nmfitemitem.do_not_estimate_yourself=true
 * </p>
 *
 * <p>
 * Item-Item models could perhaps suffer from self estimation of the item while training. With this setting you can control if the item to estimate should be excluded from the input data while training. Usually this should be true.<br>
 * Kabbur, Santosh, Xia Ning, and George Karypis. "FISM: Factored Item Similarity Models for top-n recommender systems." Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013.
 * </p>
 *
 * <p>
 * rec.nmfitemitem.adaptive_update_rules=true
 * </p>
 *
 * <p>
 * Adaptive multiplicative update rules.<br>
 * Zhang, He, Zhirong Yang, and Erkki Oja. "Adaptive multiplicative updates for projective nonnegative matrix factorization." International Conference on Neural Information Processing. Springer, Berlin, Heidelberg, 2012.<br>
 * Currently the divergence result of previous iteration is calculated in next step. So we reset the exponent of the update rules one iteration to late. Despite of this issue it works still very good.
 * </p>
 *
 * <p>
 * rec.nmfitemitem.neighbourhood_agreement=0.5
 * </p>
 *
 * <p>
 * The model assumes a complete linear calculation of the purchase probabilities from the already bought items. This seems to be a good assumption on the first sight.<br>
 * But perhaps there is a trust difference on users with a lot of bought items and users with only few bought items.<br>
 * Kabbur, Santosh, Xia Ning, and George Karypis. "FISM: Factored Item Similarity Models for top-n recommender systems." Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2013.
 * </p>
 *
 *
 * <p>
 * Item-Item Product Recommendations:
 * </p>
 *
 * <p>
 * Simply store product view history of user in browser session
 * and request the recommendation service with the users product view history (instead of the userId).
 * </p>
 *
 * <p>
 * In this algorithm the Divergence D(V || W*H*V) is minimized.
 * </p>
 *
 * <p>
 * Since the Divergence can be calculated only using non zero elements
 * the training speed is very high even on big data sets.
 * </p>
 *
 * <p>
 * Some performance optimization is done via parallel computing.
 * </p>
 *
 * <p>
 * But until now no SGD is done.
 * </p>
 *
 * <p>
 * Both matrices are updated in one iteration step at once.
 * </p>
 *
 * <p>
 * There is also no special treatment of over fitting.
 * So be careful with to much latent factors on very small training data.
 * </p>
 *
 * <p>
 * You can test the recommender with following properties:<br>
 * ( I have used movielens csv data for testing )
 * </p>
 *
 * <p>
 *  rec.recommender.class=nmfitemitem<br>
 *  rec.iterator.maximum=50<br>
 *  rec.factor.number=20<br>
 *  rec.recommender.isranking=true<br>
 *  rec.recommender.ranking.topn=10<br>
 *  <br>
 *  rec.nmfitemitem.do_not_estimate_yourself=true<br>
 *  rec.nmfitemitem.adaptive_update_rules=true<br>
 *  rec.nmfitemitem.parallelize_split_user_size=-1<br>
 *  rec.nmfitemitem.neighbourhood_agreement_loss=1<br>
 *  rec.nmfitemitem.neighbourhood_agreement_est=0<br>
 *  <br>
 *  data.model.splitter=loocv<br>
 *  data.splitter.loocv=user<br>
 *  data.convert.binarize.threshold=0<br>
 *  <br>
 *  dfs.data.dir=DATA_DIRECTORY<br>
 *  data.input.path=FILENAME.csv<br>
 *  <br>
 *  rec.eval.classes=auc,ap,arhr,hitrate,idcg,ndcg,precision,recall,rr,novelty,entropy<br>
 * </p>
 *
 * @author Daniel Velten, Karlsruhe, Germany
 */
public class NMFItemItemRecommender extends AbstractRecommender {


    private double[][] w_reconstruct = null;
    private double[][] h_analyze = null;
    private double[] b = null; // only fallback for user has 0 products bought

    private int numFactors;
    private int numIterations;

    private double divergenceFromLastStep;
    private double exponent = 0.5;
    private double neighbourhoodAgreementLoss;
    private double neighbourhoodAgreementEst;

    private int parallelizeSplitUserSize = 5000;
    private int cutoff = -1;
    private boolean doNotEstimateYourself = true;
    private boolean adaptiveUpdateRules = true;


    @Override
    protected void setup() throws LibrecException {
        super.setup();

        numFactors = conf.getInt("rec.factor.number", 15);
        numIterations = conf.getInt("rec.iterator.maximum",100);

        doNotEstimateYourself = conf.getBoolean("rec.nmfitemitem.do_not_estimate_yourself", true);
        adaptiveUpdateRules = conf.getBoolean("rec.nmfitemitem.adaptive_update_rules", true);
        parallelizeSplitUserSize = conf.getInt("rec.nmfitemitem.parallelize_split_user_size", -1);
        neighbourhoodAgreementLoss = conf.getDouble("rec.nmfitemitem.neighbourhood_agreement_loss", 1d);
        neighbourhoodAgreementEst = conf.getDouble("rec.nmfitemitem.neighbourhood_agreement_est", 0d);
        cutoff = conf.getInt("rec.nmfitemitem.cutoff", -1);

        logParameters();
        w_reconstruct = new double[numFactors][numItems];
        h_analyze = new double[numFactors][numItems];

        initMatrix(w_reconstruct);
        normFactors(w_reconstruct);
        initMatrix(h_analyze);
        normItems(h_analyze);

    }

    private void logParameters() {
        LOG.info("Using doNotEstimateYourself=" + doNotEstimateYourself);
        LOG.info("Using adaptiveUpdateRules=" + adaptiveUpdateRules);
        LOG.info("Using parallelizeSplitUserSize=" + parallelizeSplitUserSize);
        LOG.info("Using neighbourhoodAgreementLoss=" + neighbourhoodAgreementLoss);
        LOG.info("Using neighbourhoodAgreementEst=" + neighbourhoodAgreementEst);
        LOG.info("Using numFactors=" + numFactors);
        LOG.info("Using numIterations=" + numIterations);
        LOG.info("Using numUsers=" + numUsers);
        LOG.info("Using numItems=" + numItems);
        LOG.info("Using cutoff=" + cutoff);
    }

    private void normItems(double[][] h_analyze) {
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            double sum = 0;
            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                sum+=h_analyze[factorIdx][itemIdx];
            }
            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                h_analyze[factorIdx][itemIdx] /= sum;
            }
        }
    }

    private void normFactors(double[][] w_reconstruct) {
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
            double sum = 0;
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                sum+=w_reconstruct[factorIdx][itemIdx];
            }
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                w_reconstruct[factorIdx][itemIdx] /= sum;
            }
        }
    }


    private void initMatrix(double[][] m) {
        double initValue = 1d / 2d;

        Random random = new Random(123456789L);

        for (int i = 0; i < m.length; i++){
            for (int j = 0; j < m[i].length; j++){
                m[i][j] = (random.nextDouble() + 0.5) * initValue;
            }
        }

    }


    @Override
    public void trainModel() {

        if (parallelizeSplitUserSize>0 && numUsers > 2 * parallelizeSplitUserSize) {
            int availableProcessors = Runtime.getRuntime().availableProcessors();
            LOG.info("Using multithreaded. availableProcessors=" + availableProcessors);
            ExecutorService executorService = Executors.newFixedThreadPool(availableProcessors);
            for (int iter = 0; iter <= numIterations; ++iter) {
                LOG.info("Starting iteration=" + iter);
                trainMultiThreaded(executorService, iter);
            }
            executorService.shutdown();
        } else {
            LOG.info("Using singlethreaded.");
            for (int iter = 0; iter <= numIterations; ++iter) {
                LOG.info("Starting iteration=" + iter);
                trainSingleThreaded(iter);
            }
        }


        initBias();


    }


    // only for predicting items for users without previous items while evaluation phase
    private void initBias() {
        b = new double[numItems];
        double allSum =0;
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            double itemSum = trainMatrix.column(itemIdx).sum();
            b[itemIdx] = itemSum;
            allSum += itemSum;
        }
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            b[itemIdx] /= allSum;
        }
    }



    /**
     *
     * Only for storing results of the parallel executed tasks
     */
    private static class AggResult {


        private final double[][] resultNumeratorAnalyze;
        private final double[][] resultNumeratorReconstruct;
        private final double boughtItems;
        private final double sumLog;
        private final double[] countUsersBoughtItemWeighted;
        private final double[] resultDenominatorReconstruct2;

        public AggResult(double[][] resultNumeratorAnalyze, double[][] resultNumeratorReconstruct, double boughtItems, double sumLog, double[] countUsersBoughtItemWeighted, double[] resultDenominatorReconstruct2) {
            this.resultNumeratorAnalyze = resultNumeratorAnalyze;
            this.resultNumeratorReconstruct = resultNumeratorReconstruct;
            this.boughtItems = boughtItems;
            this.sumLog = sumLog;
            this.countUsersBoughtItemWeighted = countUsersBoughtItemWeighted;
            this.resultDenominatorReconstruct2 = resultDenominatorReconstruct2;
        }


    }

    /**
     *
     * Task for parallel execution.
     *
     * Executes calculations for users between 'fromUser' and 'toUser'.
     *
     */
    private class ParallelExecTask implements Callable<AggResult> {

        private final int fromUser;
        private final int toUser;

        public ParallelExecTask(int fromUser, int toUser) {
            this.fromUser = fromUser;
            this.toUser = toUser;
        }

        @Override
        public AggResult call() {
            //LOG.info("ParallelExecTask: Starting fromUser=" + fromUser + " toUser=" + toUser);
            double[][] resultNumeratorAnalyze = new double[numFactors][numItems];
            double[][] resultNumeratorReconstruct = new double[numFactors][numItems];
            double[] resultDenominatorReconstruct = new double[numFactors]; // Used in denominator

            double boughtItems = 0;
            double sumLog = 0; // only for calculating divergence for logging/debug
            double[] countUsersBoughtItemWeighted = new double[numItems]; // Used in denominator

            for (int userIdx = fromUser; userIdx < toUser; userIdx++) {
                SparseVector itemRatingsVector = trainMatrix.row(userIdx);
                int minCount = doNotEstimateYourself ? 2 : 1;
                int count = itemRatingsVector.getCount();
                if (cutoff>0 && count>cutoff){
//                    LOG.info("count=" + count);
                    continue;
                }
                if (count >= minCount) {

                    double g_est = calculateGEstimate(count);

                    double g_loss;
                    if (neighbourhoodAgreementLoss >0) {
                        g_loss = 1d / Math.pow(count - 1d, neighbourhoodAgreementLoss);
                    } else {
                        g_loss = 1d;
                    }
                    double g = g_est * g_loss;
//                    double g =((double)count)/((double)count-1);
//                    double g =1;
                    int[] itemIndices = itemRatingsVector.getIndex();
                    double[] allUserLatentFactors = predictFactors(itemIndices);

                    for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                        resultDenominatorReconstruct[factorIdx] += g * allUserLatentFactors[factorIdx];
                    }

                    double[] analyze_numerator = new double[numFactors];
                    for (int itemIdx : itemIndices) {
                        double[] thisUserLatentFactors = new double[numFactors];
                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            if (doNotEstimateYourself){
                                thisUserLatentFactors[factorIdx] = allUserLatentFactors[factorIdx] - h_analyze[factorIdx][itemIdx];
                            } else {
                                thisUserLatentFactors[factorIdx] = allUserLatentFactors[factorIdx];
                            }
                        }

                        double matrixResult = 0;
                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            matrixResult += thisUserLatentFactors[factorIdx] * w_reconstruct[factorIdx][itemIdx];
                        }
                        double estimate =  g_est * matrixResult;
                        double estimateFactor = 1d/estimate;
                        sumLog += g_loss* Math.log(estimateFactor);
                        boughtItems+=g_loss;
                        countUsersBoughtItemWeighted[itemIdx]+=g;



                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            double latent = g * thisUserLatentFactors[factorIdx];
                            resultNumeratorReconstruct[factorIdx][itemIdx] += estimateFactor * latent;

                            double numerator = estimateFactor * g * w_reconstruct[factorIdx][itemIdx];
                            analyze_numerator[factorIdx] += numerator;
                            if (doNotEstimateYourself){
                                resultNumeratorAnalyze[factorIdx][itemIdx] -= numerator;

                            }
                        }
                    }
                    for (int lItemIdx : itemIndices) {

                        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                            resultNumeratorAnalyze[factorIdx][lItemIdx] += analyze_numerator[factorIdx];
                        }
                    }
                }

            }
            return new AggResult(resultNumeratorAnalyze, resultNumeratorReconstruct, boughtItems, sumLog, countUsersBoughtItemWeighted, resultDenominatorReconstruct);
        }
    }

    private double calculateGEstimate(int count) {
        if (neighbourhoodAgreementEst >0) {
            return 1d / Math.pow(count - 1d, neighbourhoodAgreementEst);
        } else {
            return 1d;
        }
    }


    private void trainMultiThreaded(ExecutorService executorService, int iteration) {

        try {
            AggResult aggResultAll = getAggResult(executorService);
            applyAggResult(iteration, aggResultAll);
        }
        catch (InterruptedException | ExecutionException e) {
            LOG.error("", e);
            throw new IllegalStateException(e);
        }

    }

    private void trainSingleThreaded(int iteration) {
            ParallelExecTask task = new ParallelExecTask(0, numUsers);
            AggResult aggResultAll = task.call();
            applyAggResult(iteration, aggResultAll);
    }

    private void applyAggResult(int iteration, AggResult aggResultAll) {
        // Calculation of Divergence is not needed. Only for debugging/logging purpose
        double divergence = calculateDivergence(aggResultAll, iteration);

        if (adaptiveUpdateRules){
            if (iteration == 0 || divergence > divergenceFromLastStep){
                LOG.info("divergence > divergenceFromLastStep. Setting exponent to 0.5.");
                exponent = 0.5;
            } else {
                if (exponent < 1.45){
                    exponent += 0.1;
                }
                LOG.info("divergence <= divergenceFromLastStep. Exponent is now: " + exponent);
            }
            divergenceFromLastStep = divergence;
        }

        double[] wNorm = calcNormsW();
        /*
         * Multiplicative updates are done here
         *
         * Look here for explanation:
         * "Adaptive multiplicative updates for projective nonnegative matrix factorization."
         *
         */

        // We do update of both matrices at once
        // Could be changed here:

//			if (iteration % 2 ==0){
        double[][] new_w_reconstruct = updateReconstruct(aggResultAll, w_reconstruct, exponent, numFactors, numItems, h_analyze);
//				w_reconstruct = new_w_reconstruct;
//			} else {
        updateAnalyze(aggResultAll, wNorm);
//			}
        w_reconstruct = new_w_reconstruct;
    }

    private AggResult getAggResult(ExecutorService executorService) throws InterruptedException, ExecutionException {
        // Creating the parallel execution tasks
        List<ParallelExecTask> tasks = new ArrayList<>((numUsers / parallelizeSplitUserSize) + 1);
        for (int fromUser = 0; fromUser < numUsers; fromUser += parallelizeSplitUserSize) {
            int toUserExclusive = Math.min(numUsers, fromUser + parallelizeSplitUserSize);
            ParallelExecTask task = new ParallelExecTask(fromUser, toUserExclusive);
            tasks.add(task);
        }
        // Executing the tasks in parallel
        List<Future<AggResult>> results = executorService.invokeAll(tasks);

        double[][] resultNumeratorAnalyze = new double[numFactors][numItems];
        double[][] resultNumeratorReconstruct = new double[numFactors][numItems];
        double[] resultDenominatorReconstruct2 = new double[numFactors]; // Used in denominator

        double boughtItems = 0;
        double sumLog = 0; // only for calculating divergence for logging/debug
        double[] countUsersBoughtItemWeighted = new double[numItems];

        // Adding all the AggResults together..
        for (Future<AggResult> future: results) {
            AggResult result = future.get();
            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                    resultNumeratorAnalyze[factorIdx][itemIdx] += result.resultNumeratorAnalyze[factorIdx][itemIdx];
                    resultNumeratorReconstruct[factorIdx][itemIdx] += result.resultNumeratorReconstruct[factorIdx][itemIdx];
                }
                resultDenominatorReconstruct2[factorIdx] += result.resultDenominatorReconstruct2[factorIdx];
            }
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                countUsersBoughtItemWeighted[itemIdx] += result.countUsersBoughtItemWeighted[itemIdx];
            }
            boughtItems += result.boughtItems;
            sumLog += result.sumLog;
        }
        return new AggResult(resultNumeratorAnalyze, resultNumeratorReconstruct, boughtItems, sumLog, countUsersBoughtItemWeighted, resultDenominatorReconstruct2);
    }

    private double[] calcNormsW() {
        // Norms of w are not calculated in parallel (not dependent on user)
        double[] wNorm = new double[numFactors];
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
            double sum = 0;
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                sum += w_reconstruct[factorIdx][itemIdx];

            }
            wNorm[factorIdx] = sum;
        }
        return wNorm;
    }


    private void updateAnalyze(AggResult aggResult, double[] wNorm) {
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                double oldValue = h_analyze[factorIdx][itemIdx];
                double numerator = aggResult.resultNumeratorAnalyze[factorIdx][itemIdx];
                double denominator;
                if (doNotEstimateYourself){
                    denominator = aggResult.countUsersBoughtItemWeighted[itemIdx] * (wNorm[factorIdx] - w_reconstruct[factorIdx][itemIdx]);
                } else {
                    denominator = aggResult.countUsersBoughtItemWeighted[itemIdx] * wNorm[factorIdx];
                }
                double newValue = oldValue * Math.pow(numerator / denominator, exponent);

                //LOG.warn("Analyze Double.isNaN  " + numerator + " " + denominator + " " + oldValue + " " + newValue + "  " + factorIdx + "  " + itemIdx);
                if (Double.isNaN(newValue)) {
                    newValue =0;
                }
//				if (newValue<1e-16) {
//					newValue =1e-16;
//				}
                h_analyze[factorIdx][itemIdx] = newValue;
            }
        }
    }


    private static double[][] updateReconstruct(AggResult aggResultAll, double[][] w_reconstruct, double exponent, int numFactors, int numItems, double[][] h_analyze) {
        double[][] new_w_reconstruct = new double[numFactors][numItems];
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {

                double oldValue = w_reconstruct[factorIdx][itemIdx];
                double numerator = aggResultAll.resultNumeratorReconstruct[factorIdx][itemIdx];
                double denominatorDiff = aggResultAll.countUsersBoughtItemWeighted[itemIdx]*h_analyze[factorIdx][itemIdx];
                double denominator = aggResultAll.resultDenominatorReconstruct2[factorIdx];
                double newValue = oldValue * Math.pow(numerator / (denominator - denominatorDiff), exponent);

//		if (Double.isNaN(newValue)) {
//			LOG.warn("Double.isNaN  " + numerator + " " + denominator +" " + denominator2 + " " + oldValue + " " + newValue);
//		}
//		if (newValue<1e-16) {
//			newValue =1e-16;
//		}
                new_w_reconstruct[factorIdx][itemIdx] = newValue;
            }
        }
        return new_w_reconstruct;
    }


    private double calculateDivergence(AggResult aggResultAll, int iteration) {
        double sumAllEstimate = 0;
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
            for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {

                double denominatorDiff = aggResultAll.countUsersBoughtItemWeighted[itemIdx]*h_analyze[factorIdx][itemIdx];
                double denominator = aggResultAll.resultDenominatorReconstruct2[factorIdx];
                double newValue = denominator - denominatorDiff;

                sumAllEstimate += w_reconstruct[factorIdx][itemIdx] * newValue;
            }
        }

        double divergence = aggResultAll.sumLog- aggResultAll.boughtItems + sumAllEstimate;
        LOG.info("Divergence (before iteration " + iteration +")=" + divergence + "  sumLog=" + aggResultAll.sumLog + "  countAll=" + aggResultAll.boughtItems + "  sumAllEstimate=" + sumAllEstimate);
        //LOG.info("Divergence (before iteration " + iteration +")=" + divergence);

        return divergence;
    }

    private double predict(SparseVector itemRatingsVector, int itemIdx) {
        double sum = 0;
        for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {

            sum += w_reconstruct[factorIdx][itemIdx] * predictFactor(itemRatingsVector, factorIdx);

        }

        return sum;
    }

    private double predictFactor(SparseVector itemRatingsVector, int factorIdx) {
        double sum = 0;
        for (int itemIdx : itemRatingsVector.getIndex()) {
            sum += w_reconstruct[factorIdx][itemIdx];
        }
        return sum;
    }

    private double[] predictFactors(int[] itemIndices) {
        double[] latentFactors = new double[numFactors];
        for (int itemIdx : itemIndices) {
            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                latentFactors[factorIdx] += h_analyze[factorIdx][itemIdx];
            }
        }
        return latentFactors;
    }

    /*
     * This is not fast if you call for each item from outside
     * Calculate factors first and then calculate with factors the prediction of each item
     */
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        SparseVector itemRatingsVector = trainMatrix.row(userIdx);

        return predict(itemRatingsVector, itemIdx);
    }

    /*
     * This method is overridden only for performance reasons.
     *
     * Calculate all item ratings at once for one user has much better performance than for each item user combination alone.
     *
     * Effect is significant on big data
     */
    @Override
    protected RecommendedList recommendRank() throws LibrecException {
        RecommendedItemList recommendedList = new RecommendedItemList(numUsers - 1, numUsers);

        LOG.info("Calculating RecommendedList for " + numUsers + " users");
        Comparator<ItemValue> comparator = new Comparator<ItemValue>() {
            @Override
            public int compare(ItemValue o1, ItemValue o2) {
                return Double.compare(o1.value, o2.value);
            }

        };
        for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
            if (userIdx%100000==0) {
                LOG.info("At user " + userIdx);
            }
            SparseVector itemRatingsVector = trainMatrix.row(userIdx);
            TreeSet<ItemValue> sorted = new TreeSet<>(comparator);
            int count = itemRatingsVector.size();
            if (count ==0){
                for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                    double rating = b[itemIdx];
                    addToSet(sorted, itemIdx, rating);
                }
            } else {

                double[] thisUserLatentFactors = predictFactors(itemRatingsVector.getIndex());


                for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
                    if (itemRatingsVector.contains(itemIdx)) {
                        continue;
                    }
                    double predictRating = 0;
                    for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                        predictRating += thisUserLatentFactors[factorIdx] * w_reconstruct[factorIdx][itemIdx];
                    }
                    if (Double.isNaN(predictRating)) {
                        continue;
                    }
                    double g_est=calculateGEstimate(count);
                    double predictRating2=g_est * predictRating;
                    addToSet(sorted, itemIdx, predictRating2);
                }
            }
            ArrayList<ItemEntry<Integer, Double>> list = new ArrayList<>(sorted.size());
            Iterator<ItemValue> it = sorted.descendingIterator();
            while (it.hasNext()) {
                ItemValue entry =  it.next();
                list.add(new ItemEntry<Integer, Double>(entry.itemIdx, entry.value));
            }
            recommendedList.setItemIdxList(userIdx, list);
        }

        if(recommendedList.size()==0){
            throw new IndexOutOfBoundsException("No item is recommended, there is something error in the recommendation algorithm! Please check it!");
        }

        logParameters();

        return recommendedList;
    }

    private static class ItemValue {
        private final int itemIdx;
        private final double value;

        public ItemValue(int itemIdx, double value) {
            this.itemIdx = itemIdx;
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ItemValue that = (ItemValue) o;
            return itemIdx == that.itemIdx &&
                    Double.compare(that.value, value) == 0;
        }

        @Override
        public int hashCode() {

            return Objects.hash(itemIdx, value);
        }

        @Override
        public String toString() {
            return "ItemValue{" +
                    "itemIdx=" + itemIdx +
                    ", value=" + value +
                    '}';
        }
    }

    private void addToSet(TreeSet<ItemValue> sorted, int itemIdx, double rating) {
        if (sorted.isEmpty() || sorted.first().value < rating){
            sorted.add(new ItemValue(itemIdx, rating));
            if (sorted.size()>topN) {
                sorted.pollFirst();
            }
        }
    }


    @Override
    public void saveModel(String directoryPath) {
        File dir = new File(directoryPath);
        dir.mkdir();

        try{
            File wFile = new File(dir, "w_reconstruct.csv");
            LOG.info("Writing matrix w_reconstruct to file=" + wFile.getAbsolutePath());
            saveMatrix(wFile, w_reconstruct);
            File hFile = new File(dir, "h_analyze.csv");
            LOG.info("Writing matrix h_analyze to file=" + hFile.getAbsolutePath());
            saveMatrix(hFile, h_analyze);
        } catch (Exception e) {
            LOG.error("Could not save model", e);
        }
    }


    private void saveMatrix(File file, double[][] matrix) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(file));
        writer.write("\"item_id\"");
        for (int i = 0; i < numFactors; i++) {
            writer.write(',');
            writer.write("\"factor");
            writer.write(Integer.toString(i));
            writer.write("\"");
        }
        writer.write("\r\n");
        BiMap<Integer, String> items = itemMappingData.inverse();
        for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
            writer.write('\"');
            writer.write(items.get(itemIdx));
            writer.write('\"');
            for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                writer.write(',');
                writer.write(Double.toString(matrix[factorIdx][itemIdx]));
            }
            writer.write("\r\n");
        }
        writer.close();
    }
}
