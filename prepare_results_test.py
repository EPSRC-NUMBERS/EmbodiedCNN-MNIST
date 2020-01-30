'''
Calculates the statistics for the Convolutional models
'''

import numpy as np
import os
from scipy.stats import ttest_ind_from_stats, kruskal
from shutil import rmtree

folder ='./Logs/'

folder_data = folder+'histories/test/'
folder_output = folder+'output/test/'

if not os.path.exists(folder_data):
	os.makedirs(folder_data)
	os.makedirs(folder_data+'accuracy')
	os.makedirs(folder_data+'likelihood')

if not os.path.exists(folder_output):
	os.makedirs(folder_output)
	os.makedirs(folder_output+'accuracy')
	os.makedirs(folder_output+'likelihood')
	os.makedirs(folder_output+'stats')
	os.makedirs(folder_output+'comparisons')

N=25 #number of epochs
reps=5

ssplit = np.zeros((7,1)) # number of examples
nsplit = ssplit.shape[0]
ntest = 10000
full_acc_max = np.zeros(shape=(4,4*nsplit))
full_like_max = np.zeros(shape=(4,4*nsplit))
pvalues_acc = np.zeros((nsplit*2,6))
pvalues_like = np.zeros((nsplit*2,6))

kruskal_acc = np.zeros((nsplit*2,1))
kruskal_like = np.zeros((nsplit*2,1))

final = np.zeros(shape=(4,nsplit,reps))

for i in range(nsplit):
	print('Split = ',i)


	accuracy1 = np.zeros(shape=(N,reps))
	loss1 = np.zeros(shape=(N,reps))
	likelihood1 = np.zeros(shape=(N,reps))
	full_data1 = np.zeros(shape=(N,8))

	accuracy2 = np.zeros(shape=(N,reps))
	loss2 = np.zeros(shape=(N,reps))
	likelihood2 = np.zeros(shape=(N,reps))
	full_data2 = np.zeros(shape=(N,8))

	accuracy3 = np.zeros(shape=(N,reps))
	loss3 = np.zeros(shape=(N,reps))
	likelihood3 = np.zeros(shape=(N,reps))
	full_data3 = np.zeros(shape=(N,8))

	accuracy4 = np.zeros(shape=(N,reps))
	loss4 = np.zeros(shape=(N,reps))
	likelihood4 = np.zeros(shape=(N,reps))
	full_data4 = np.zeros(shape=(N,8))

	for k in range(reps): # for all the repetitions
		#Le-Net5
		my_data1 = np.genfromtxt(folder+str(k)+'/training_lenet5_'+"{:03d}".format(i)+'.log',delimiter=',')
		loss4[:,k] = my_data1[1:N+1,3]
		accuracy4[:,k] = (my_data1[1:N+1,1]*ssplit[i] + my_data1[1:N+1,5]*ntest)/(ssplit[i]+ntest)# cuts the header, line 0
		likelihood4[:,k] = my_data1[1:N+1,6]

		# #fingers-termometer representation
		my_data2 = np.genfromtxt(folder+str(k)+'/training_class0_conv2d'+"{:03d}".format(i)+'.log',delimiter=',')
		loss2[:,k] = my_data2[1:N+1,3]
		accuracy2[:,k] = (my_data2[1:N+1,1]*ssplit[i] + my_data2[1:N+1,8]*ntest)/(ssplit[i]+ntest)
		likelihood2[:,k] = my_data2[1:N+1,9]

		# #random
		my_data3 = np.genfromtxt(folder+str(k)+'/training_robot1_conv2d'+"{:03d}".format(i)+'.log',delimiter=',')
		loss3[:,k] = my_data3[1:N+1,3]
		accuracy3[:,k] = (my_data3[1:N+1,1]*ssplit[i] + my_data3[1:N+1,8]*ntest)/(ssplit[i]+ntest)
		likelihood3[:,k] = my_data3[1:N+1:,8]

		#robot
		my_data4 = np.genfromtxt(folder+str(k)+'/training_robotP_conv2d'+"{:03d}".format(i)+'.log',delimiter=',')
		loss1[:,k] = my_data4[1:N+1,3]
		accuracy1[:,k] = (my_data4[1:N+1,1]*ssplit[i] + my_data4[1:N+1,8]*ntest)/(ssplit[i]+ntest)
		likelihood1[:,k] = my_data4[1:N+1,9]

		for g in range(1,N):
			if loss1[g,k] >loss1[g-1,k]:
				loss1[g,k] = loss1[g-1,k]
				accuracy1[g,k] = accuracy1[g-1,k]
			if loss2[g,k] > loss2[g-1,k]:
				loss2[g,k] = loss2[g-1,k]
				accuracy2[g,k] = accuracy2[g-1,k]
			if loss3[g,k] > loss3[g-1,k]:
				loss3[g,k] = loss3[g-1,k]
				accuracy3[g,k] = accuracy3[g-1,k]
			if loss4[g,k] > loss4[g-1,k]:
				loss4[g,k] = loss4[g-1,k]
				accuracy4[g,k] = accuracy4[g-1,k]

	np.savetxt(folder_data+'accuracy/accuracy_mnist_base'+"{:03d}".format(i)+'.csv', accuracy1, delimiter=",")
	np.savetxt(folder_data+'accuracy/accuracy_mnist_nummag'+"{:03d}".format(i)+'.csv', accuracy2, delimiter=",")
	np.savetxt(folder_data+'accuracy/accuracy_mnist_random'+"{:03d}".format(i)+'.csv', accuracy3, delimiter=",")
	np.savetxt(folder_data+'accuracy/accuracy_mnist_robot'+"{:03d}".format(i)+'.csv', accuracy4, delimiter=",")
	np.savetxt(folder_data+'likelihood/likelihood_mnist_base'+"{:03d}".format(i)+'.csv', likelihood1, delimiter=",")
	np.savetxt(folder_data+'likelihood/likelihood_mnist_nummag'+"{:03d}".format(i)+'.csv', likelihood2, delimiter=",")
	np.savetxt(folder_data+'likelihood/likelihood_mnist_random'+"{:03d}".format(i)+'.csv', likelihood3, delimiter=",")
	np.savetxt(folder_data+'likelihood/likelihood_mnist_robot'+"{:03d}".format(i)+'.csv', likelihood4, delimiter=",")

	full_data1[:,0] = np.mean(accuracy1,axis=1)
	full_data1[:,1] = np.std(accuracy1,axis=1)
	full_data1[:,2] = np.median(accuracy1,axis=1)
	full_data1[:,3] = np.max(accuracy1,axis=1)
	full_data1[:,4] = np.mean(likelihood1,axis=1)
	full_data1[:,5] = np.std(likelihood1,axis=1)
	full_data1[:,6] = np.median(likelihood1,axis=1)
	full_data1[:,7] = np.max(likelihood1,axis=1)
	
	np.savetxt(folder_output+'stats/results_mnist_classify'+"{:03d}".format(i)+'.csv', full_data1, delimiter=",")

	full_data2[:,0] = np.mean(accuracy2,axis=1)
	full_data2[:,1] = np.std(accuracy2,axis=1)
	full_data2[:,2] = np.median(accuracy2,axis=1)
	full_data2[:,3] = np.max(accuracy2,axis=1)
	full_data2[:,4] = np.mean(likelihood2,axis=1)
	full_data2[:,5] = np.std(likelihood2,axis=1)
	full_data2[:,6] = np.median(likelihood2,axis=1)
	full_data2[:,7] = np.max(likelihood2,axis=1)

	np.savetxt(folder_output+'stats/results_mnist_nummag'+"{:03d}".format(i)+'.csv', full_data2, delimiter=",")

	full_data3[:,0] = np.mean(accuracy3,axis=1)
	full_data3[:,1] = np.std(accuracy3,axis=1)
	full_data3[:,2] = np.median(accuracy3,axis=1)
	full_data3[:,3] = np.max(accuracy3,axis=1)
	full_data3[:,4] = np.mean(likelihood3,axis=1)
	full_data3[:,5] = np.std(likelihood3,axis=1)
	full_data3[:,6] = np.median(likelihood3,axis=1)
	full_data3[:,7] = np.max(likelihood3,axis=1)

	np.savetxt(folder_output+'stats/results_mnist_random'+"{:03d}".format(i)+'.csv', full_data3, delimiter=",")

	full_data4[:,0] = np.mean(accuracy4,axis=1)
	full_data4[:,1] = np.std(accuracy4,axis=1)
	full_data4[:,2] = np.median(accuracy4,axis=1)
	full_data4[:,3] = np.max(accuracy4,axis=1)
	full_data4[:,4] = np.mean(likelihood4,axis=1)
	full_data4[:,5] = np.std(likelihood4,axis=1)
	full_data4[:,6] = np.median(likelihood4,axis=1)
	full_data4[:,7] = np.max(likelihood4,axis=1)

	np.savetxt(folder_output+'stats/results_mnist_robot'+"{:03d}".format(i)+'.csv', full_data4, delimiter=",")

	EQVAR = False
	#ttest for accuracy
	ttest12a = ttest_ind_from_stats(mean1=full_data1[:N,0],std1=full_data1[:N,1],nobs1=reps,mean2=full_data2[:,0],std2=full_data2[:,1],nobs2=reps,equal_var=EQVAR)
	ttest13a = ttest_ind_from_stats(mean1=full_data1[:N,0],std1=full_data1[:N,1],nobs1=reps,mean2=full_data3[:,0],std2=full_data3[:,1],nobs2=reps,equal_var=EQVAR)
	ttest14a = ttest_ind_from_stats(mean1=full_data1[:N,0],std1=full_data1[:N,1],nobs1=reps,mean2=full_data4[:,0],std2=full_data4[:,1],nobs2=reps,equal_var=EQVAR)
	ttest23a = ttest_ind_from_stats(mean1=full_data2[:N,0],std1=full_data2[:N,1],nobs1=reps,mean2=full_data3[:,0],std2=full_data3[:,1],nobs2=reps,equal_var=EQVAR)
	ttest24a = ttest_ind_from_stats(mean1=full_data2[:N,0],std1=full_data2[:N,1],nobs1=reps,mean2=full_data4[:,0],std2=full_data4[:,1],nobs2=reps,equal_var=EQVAR)
	ttest34a = ttest_ind_from_stats(mean1=full_data3[:N,0],std1=full_data3[:N,1],nobs1=reps,mean2=full_data4[:,0],std2=full_data4[:,1],nobs2=reps,equal_var=EQVAR)

	#ttest for likelihood
	ttest12l = ttest_ind_from_stats(mean1=full_data1[:N,4],std1=full_data1[:N,5],nobs1=reps,mean2=full_data2[:,4],std2=full_data2[:,5],nobs2=reps,equal_var=EQVAR)
	ttest13l = ttest_ind_from_stats(mean1=full_data1[:N,4],std1=full_data1[:N,5],nobs1=reps,mean2=full_data3[:,4],std2=full_data3[:,5],nobs2=reps,equal_var=EQVAR)
	ttest14l = ttest_ind_from_stats(mean1=full_data1[:N,4],std1=full_data1[:N,5],nobs1=reps,mean2=full_data4[:,4],std2=full_data4[:,5],nobs2=reps,equal_var=EQVAR)
	ttest23l = ttest_ind_from_stats(mean1=full_data2[:N,4],std1=full_data2[:N,5],nobs1=reps,mean2=full_data3[:,4],std2=full_data3[:,5],nobs2=reps,equal_var=EQVAR)
	ttest24l = ttest_ind_from_stats(mean1=full_data2[:N,4],std1=full_data2[:N,5],nobs1=reps,mean2=full_data4[:,4],std2=full_data4[:,5],nobs2=reps,equal_var=EQVAR)
	ttest34l = ttest_ind_from_stats(mean1=full_data3[:N,4],std1=full_data3[:N,5],nobs1=reps,mean2=full_data4[:,4],std2=full_data4[:,5],nobs2=reps,equal_var=EQVAR)


	comparisons_a = np.zeros(shape=(N,14))
	comparisons_a[:,0] = full_data1[:N,2]
	comparisons_a[:,1] = full_data2[:N,2]
	comparisons_a[:,2] = full_data3[:N,2]
	comparisons_a[:,3] = full_data4[:N,2]
	comparisons_a[:,4] = full_data1[:N,1]
	comparisons_a[:,5] = full_data2[:N,1]
	comparisons_a[:,6] = full_data3[:N,1]
	comparisons_a[:,7] = full_data4[:N,1]
	comparisons_a[:,8] = ttest12a.pvalue
	comparisons_a[:,9] = ttest13a.pvalue
	comparisons_a[:,10] = ttest14a.pvalue
	comparisons_a[:,11] = ttest23a.pvalue
	comparisons_a[:,12] = ttest24a.pvalue
	comparisons_a[:,13] = ttest34a.pvalue

	comparisons_l = np.zeros(shape=(N,14))
	comparisons_l[:,0] = full_data1[:N,4]
	comparisons_l[:,1] = full_data2[:N,4]
	comparisons_l[:,2] = full_data3[:N,4]
	comparisons_l[:,3] = full_data4[:N,4]
	comparisons_l[:,4] = full_data1[:N,5]
	comparisons_l[:,5] = full_data2[:N,5]
	comparisons_l[:,6] = full_data3[:N,5]
	comparisons_l[:,7] = full_data4[:N,5]
	comparisons_l[:,8] = ttest12l.pvalue
	comparisons_l[:,9] = ttest13l.pvalue
	comparisons_l[:,10] = ttest14l.pvalue
	comparisons_l[:,11] = ttest23l.pvalue
	comparisons_l[:,12] = ttest24l.pvalue
	comparisons_l[:,13] = ttest34l.pvalue


	summary_a = np.zeros(shape=(N,6))
	summary_a[:,0] = (full_data2[:,0]-full_data1[:N,0])*(ttest12a.pvalue<0.01)
	summary_a[:,1] = (full_data3[:,0]-full_data1[:N,0])*(ttest13a.pvalue<0.01)
	summary_a[:,2] = (full_data4[:,0]-full_data1[:N,0])*(ttest14a.pvalue<0.01)
	summary_a[:,3] = (full_data2[:,0]-full_data3[:N,0])*(ttest23a.pvalue<0.01)
	summary_a[:,4] = (full_data4[:,0]-full_data2[:N,0])*(ttest24a.pvalue<0.01)
	summary_a[:,5] = (full_data4[:,0]-full_data3[:N,0])*(ttest34a.pvalue<0.01)


	summary_l = np.zeros(shape=(N,6))
	summary_l[:,0] = (full_data2[:,4]-full_data1[:N,4])*(ttest12a.pvalue<0.01)
	summary_l[:,1] = (full_data3[:,4]-full_data1[:N,4])*(ttest13a.pvalue<0.01)
	summary_l[:,2] = (full_data4[:,4]-full_data1[:N,4])*(ttest14a.pvalue<0.01)
	summary_l[:,3] = (full_data2[:,4]-full_data3[:N,4])*(ttest23a.pvalue<0.01)
	summary_l[:,4] = (full_data4[:,4]-full_data2[:N,4])*(ttest24a.pvalue<0.01)
	summary_l[:,5] = (full_data4[:,4]-full_data3[:N,4])*(ttest34a.pvalue<0.01)

	np.savetxt(folder_output+'accuracy/results_acc_comparisons'+"{:03d}".format(i)+'.csv', comparisons_a, delimiter=",", header="baseline,termometer,random,robot,std1,std2,std3,std4,p12,p13,p14,p23,p24,p34")
	np.savetxt(folder_output+'likelihood/results_like_comparisons'+"{:03d}".format(i)+'.csv', comparisons_l, delimiter=",",  header="baseline,termometer,random,robot,std1,std2,std3,std4,p12,p13,p14,p23,p24,p34")
	np.savetxt(folder_output+'accuracy/results_acc_summary'+"{:03d}".format(i)+'.csv', summary_a, delimiter=",", header='term/base,rand/base,robot/base,term/random,robot/term,robot/random')
	np.savetxt(folder_output+'likelihood/results_like_summary'+"{:03d}".format(i)+'.csv', summary_l, delimiter=",", header='term/base,rand/base,robot/base,term/random,robot/term,robot/random')


	min_loss1 = np.zeros((reps,1))
	for l in range(reps):
		min_loss1[l] = np.argmin(loss1[:,l])

	print(np.min(min_loss1))

	final[0,i,:] = np.take(full_data1[:,0],min_loss1.astype(int)).reshape(reps)

	min_loss2 = np.zeros((reps,1))
	for l in range(reps):
		min_loss2[l] = np.argmin(loss2[:,l])

	print(np.min(min_loss2))	

	final[1,i,:] = np.take(full_data2[:,0],min_loss2.astype(int)).reshape(reps)

	min_loss3 = np.zeros((reps,1))
	for l in range(reps):
		min_loss3[l] = np.argmin(loss3[:,l])

	final[2,i,:] = np.take(full_data3[:,0],min_loss3.astype(int)).reshape(reps)

	print(np.min(min_loss3))	

	min_loss4 = np.zeros((reps,1))
	for l in range(reps):
		min_loss4[l] = np.argmin(loss4[:,l])

	print(np.min(min_loss4))	

	final[3,i,:] = np.take(full_data4[:,0],min_loss4.astype(int)).reshape(reps)

	acc_max = np.zeros((4,6))
	for s in range(4):
		a = N-1#np.argmax(comparisons_a[:,s])
		#a = np.where(comparisons_a[:,s]==find_nearest(comparisons_a[:,s],np.median(comparisons_a[14:,s])))
		#a = a[0][0]
		acc_max[s,0] = comparisons_a[a,0+s]
		acc_max[s,1] = comparisons_a[a,4+s]

	#s=3
	#acc_max[s,0] = np.mean(final[s,i,:])
	#acc_max[s,1] = np.std(final[s,i,:])

	for s in range(4):
		for v in range(4):
			acc_max[s,2+v] = ttest_ind_from_stats(mean1=acc_max[s,0],std1=acc_max[s,1],nobs1=reps,mean2=acc_max[v,0],std2=acc_max[v,1],nobs2=reps,equal_var=EQVAR).pvalue


	comparisons_a[N-1,8] = ttest_ind_from_stats(mean1=acc_max[0,0],std1=acc_max[0,1],nobs1=reps,mean2=acc_max[1,0],std2=acc_max[1,1],nobs2=reps,equal_var=EQVAR).pvalue
	comparisons_a[N-1,9] = ttest_ind_from_stats(mean1=acc_max[0,0],std1=acc_max[0,1],nobs1=reps,mean2=acc_max[2,0],std2=acc_max[2,1],nobs2=reps,equal_var=EQVAR).pvalue
	comparisons_a[N-1,10] = ttest_ind_from_stats(mean1=acc_max[0,0],std1=acc_max[0,1],nobs1=reps,mean2=acc_max[3,0],std2=acc_max[3,1],nobs2=reps,equal_var=EQVAR).pvalue
	comparisons_a[N-1,11] = ttest_ind_from_stats(mean1=acc_max[1,0],std1=acc_max[1,1],nobs1=reps,mean2=acc_max[2,0],std2=acc_max[2,1],nobs2=reps,equal_var=EQVAR).pvalue
	comparisons_a[N-1,12] = ttest_ind_from_stats(mean1=acc_max[1,0],std1=acc_max[1,1],nobs1=reps,mean2=acc_max[3,0],std2=acc_max[3,1],nobs2=reps,equal_var=EQVAR).pvalue
	comparisons_a[N-1,13] = ttest_ind_from_stats(mean1=acc_max[2,0],std1=acc_max[2,1],nobs1=reps,mean2=acc_max[3,0],std2=acc_max[3,1],nobs2=reps,equal_var=EQVAR).pvalue

	full_acc_max[:,i] = comparisons_a[0,:4].transpose()
	full_acc_max[:,i+nsplit] = comparisons_a[0,4:8].transpose()
	full_acc_max[:,i+(nsplit*2)] = acc_max[:,0]
	full_acc_max[:,i+(nsplit*3)] = acc_max[:,1]

	np.savetxt(folder_output+'comparisons/max_comparisons'+"{:03d}".format(i)+'.csv', acc_max, delimiter=",", header="max,std,px1,px2,px3,px4")

	like_max = np.zeros((4,6))
	for s in range(4):
		a = np.argmax(comparisons_l[:,s])
		like_max[s,0] = comparisons_l[a,0+s]
		like_max[s,1] = comparisons_l[a,4+s]

	for s in range(4):
		for v in range(4):
			like_max[s,2+v] = ttest_ind_from_stats(mean1=like_max[s,0],std1=like_max[s,1],nobs1=reps,mean2=like_max[v,0],std2=like_max[v,1],nobs2=reps,equal_var=EQVAR).pvalue

	full_like_max[:,i] = comparisons_l[0,:4].transpose()
	full_like_max[:,i+nsplit] = comparisons_l[0,4:8].transpose()
	full_like_max[:,i+(nsplit*2)] = like_max[:,0]
	full_like_max[:,i+(nsplit*3)] = like_max[:,1]

	np.savetxt(folder_output+'comparisons/like_comparisons'+"{:03d}".format(i)+'.csv', like_max, delimiter=",", header="max,std,px1,px2,px3,px4")

	pvalues_acc[[i,i+nsplit],:] = comparisons_a[[0,N-1],8:]
	pvalues_like[[i,i+nsplit],:] = comparisons_l[[0,N-1],8:]

	kruskal_acc[i] = kruskal(accuracy2[0,:],accuracy3[0,:],accuracy4[0,:])[1]
	kruskal_acc[i+nsplit] = kruskal(accuracy2[N-1,:],accuracy3[N-1,:],accuracy4[N-1,:])[1]

	kruskal_like[i] = kruskal(likelihood2[0,:],likelihood3[0,:],likelihood4[0,:])[1]
	kruskal_like[i+nsplit] = kruskal(likelihood2[N-1,:],likelihood3[N-1,:],likelihood4[N-1,:])[1]

full_acc_max = full_acc_max.transpose()
full_like_max = full_like_max.transpose()

cohensd_acc = np.zeros((nsplit*2,3))
glassD_acc = np.zeros((nsplit*2,3))
for d in range(1,4):
	cohensd_acc[:nsplit,d-1] = (full_acc_max[:nsplit,d]-full_acc_max[:nsplit,0])/(np.sqrt((np.square(full_acc_max[nsplit:nsplit*2,d])+np.square(full_acc_max[nsplit:nsplit*2,0]))/2))
	cohensd_acc[nsplit:,d-1] = (full_acc_max[nsplit*2:nsplit*3,d]-full_acc_max[nsplit*2:nsplit*3,0])/(np.sqrt((np.square(full_acc_max[nsplit*3:nsplit*4,d])+np.square(full_acc_max[nsplit*3:nsplit*4,0]))/2))
	glassD_acc[:nsplit,d-1] = (full_acc_max[:nsplit,d]-full_acc_max[:nsplit,0])/(full_acc_max[nsplit:nsplit*2,0])
	glassD_acc[nsplit:,d-1] = (full_acc_max[nsplit*2:nsplit*3,d]-full_acc_max[nsplit*2:nsplit*3,0])/(full_acc_max[nsplit*3:nsplit*4,0])

correction = ((reps-3)/(reps-2.25))*np.sqrt((reps-2.0)/reps)

cohensd_acc *= correction

cohensd_like = np.zeros((nsplit*2,3))
glassD_like = np.zeros((nsplit*2,3))
for d in range(1,4):
	cohensd_like[:nsplit,d-1] = (full_like_max[:nsplit,d]-full_like_max[:nsplit,0])/(np.sqrt((np.square(full_like_max[nsplit:nsplit*2,d])+np.square(full_like_max[nsplit:nsplit*2,0]))/2))
	cohensd_like[nsplit:,d-1] = (full_like_max[nsplit*2:nsplit*3,d]-full_like_max[nsplit*2:nsplit*3,0])/(np.sqrt((np.square(full_like_max[nsplit*3:nsplit*4,d])+np.square(full_like_max[nsplit*3:nsplit*4,0]))/2))
	glassD_like[:nsplit,d-1] = (full_like_max[:nsplit,d]-full_like_max[:nsplit,0])/(full_like_max[nsplit:nsplit*2,0])
	glassD_like[nsplit:,d-1] = (full_like_max[nsplit*2:nsplit*3,d]-full_like_max[nsplit*2:nsplit*3,0])/(full_like_max[nsplit*3:nsplit*4,0])

cohensd_like *= correction

acc_summary = np.zeros((nsplit*2,11))
acc_summary[:nsplit,[0,2,5,8]] = full_acc_max[:nsplit,:]
acc_summary[:nsplit,[1,3,6,9]] = full_acc_max[nsplit:nsplit*2,:]
acc_summary[nsplit:,[0,2,5,8]] = full_acc_max[nsplit*2:nsplit*3,:]
acc_summary[nsplit:,[1,3,6,9]] = full_acc_max[nsplit*3:nsplit*4,:]
acc_summary[:nsplit,[4,7,10]] = cohensd_acc[:nsplit,:]
acc_summary[nsplit:,[4,7,10]] = cohensd_acc[nsplit:,:]
#acc_summary[:nsplit,[4,7,10]] = glassD_acc[:nsplit,:]
#acc_summary[nsplit:,[4,7,10]] = glassD_acc[nsplit:,:]


like_summary = np.zeros((nsplit*2,11))
like_summary[:nsplit,[0,2,5,8]] = full_like_max[:nsplit,:]
like_summary[:nsplit,[1,3,6,9]] = full_like_max[nsplit:nsplit*2,:]
like_summary[nsplit:,[0,2,5,8]] = full_like_max[nsplit*2:nsplit*3,:]
like_summary[nsplit:,[1,3,6,9]] = full_like_max[nsplit*3:nsplit*4,:]
like_summary[:nsplit,[4,7,10]] = cohensd_like[:nsplit,:]
like_summary[nsplit:,[4,7,10]] = cohensd_like[nsplit:,:]
#like_summary[:nsplit,[4,7,10]] = glassD_like[:nsplit,:]
#like_summary[nsplit:,[4,7,10]] = glassD_like[nsplit:,:]

np.savetxt(folder_output+'acc_summary.csv', np.concatenate((acc_summary, pvalues_acc,kruskal_acc),axis=1), delimiter=",", 
	header="base,std,class,std,cd,random,std,cd,robot,std,cd,p12,p13,p14,p23,p24,p34,kruskal")
np.savetxt(folder_output+'like_summary.csv', np.concatenate((like_summary,pvalues_like,kruskal_like),axis=1), delimiter=",", 
	header="base,std,class,std,cd,random,std,cd,robot,std,cd,p12,p13,p14,p23,p24,p34,kruskal")