import streamlit as st
import h5py
import numpy as np
from unet.unet import UNet1D
import torch
from utils import read_process_mat_file, sigmoid
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from stqdm import stqdm
from sklearn import metrics
import streamlit.components.v1 as components
import base64
import json
import scipy
import os 

st.title("Seizure Prediction")

uploaded_file = st.file_uploader("Upload a .mat file with raw ECG data")
uploaded_model = st.file_uploader("Upload a trained UNET1D model (.pt file)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

if (uploaded_file is not None) and (uploaded_model is not None):
    st.write('Processing raw ECG data')
    
    ecog_list, label_list, ecog_downs, label_downs = read_process_mat_file(uploaded_file)
    
    st.write('Finished processing raw ECG data')
    st.write(f'Number of 20 second segments in the downsampled data: {len(ecog_list)}')
    # st.write("Filename: ", uploaded_file.name)
    # st.line_chart(ecog_downs.numpy()[0][:10000])

    model = UNet1D(
        normalization='batch',
        preactivation=False,
        residual=True,
        out_classes=1,
        num_encoding_blocks=5,
        encoder_kernel_sizes=[5,5,5,5,5],
        decoder_kernel_sizes=[3]*4,
    )
    
    if uploaded_model is not None:
        model.load_state_dict(torch.load(uploaded_model, weights_only=False))
    model = model.to(device)
    model.eval()
    
    # ind = int(st.number_input(f'Insert a segment number of {len(ecog_list)}', value=0, placeholder="Type a number...", min_value=0, max_value=len(ecog_list)))
    
    with st.form("my_form"):
        st.write("Seizure prediction using UNET1D on individual segments")
        # slider_val = st.slider("Form slider")
        ind = int(st.number_input(f'Insert a segment number of {len(ecog_list)}', value=0, placeholder="Type a number...", min_value=0, max_value=len(ecog_list)))
        # checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            with torch.no_grad():
                outs_np=model(ecog_list[ind].unsqueeze(0).unsqueeze(0).to(device))[0][0].cpu().detach().numpy()

            prob_mask=sigmoid(outs_np.astype(np.float64))
            pred_mask = (prob_mask > 0.5).astype(int).tolist()
            
            plotdf=pd.DataFrame()
            plotdf['ecog']=ecog_list[ind].tolist()
            plotdf['preds']=pred_mask
            plotdf['probs']=prob_mask
            plotdf['label']=label_list[ind].tolist()
            

            st.write(f"F1 Score: {f1_score(plotdf['label'], plotdf['preds'])}")
            st.write(f"Precision Score: {precision_score(plotdf['label'], plotdf['preds'])}")
            st.write(f"Recall Score: {recall_score(plotdf['label'], plotdf['preds'])}")

            # f, ax=plt.subplots(1, figsize=(20,8))
            # sns.lineplot(data=plotdf, x=plotdf.index, y="ecog", hue="label", ax=ax)
            # st.pyplot(f)
            
            color_discrete_map = {1:'red',0:'blue'}
            fig1 = px.line(plotdf, x=plotdf.index, y="ecog", color='preds', color_discrete_map=color_discrete_map, title='Predicted Label')
            fig2 = px.line(plotdf, x=plotdf.index, y="ecog", color='label', color_discrete_map=color_discrete_map, title="True Label")
            # fig.show()

            
            # fig.show()
            
            fig3 = px.line(plotdf, x=plotdf.index, y="probs", title='Probs')
            fig4 = px.line(plotdf, x=plotdf.index, y="label", title='Labels')
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(fig4, use_container_width=True)
            
    # @st.cache_data
    def predict_whole_data(_model, _ecog_list, _label_list):
        preds=[]
        probs=[]
        testecoglist=[]
        testlabellist=[]
        for i, (testecog, testlabel) in enumerate(stqdm(zip(_ecog_list, _label_list), total=len(_ecog_list), desc='Predicting on whole data')):
            with torch.no_grad():
                outputs = _model(testecog.unsqueeze(0).unsqueeze(0).to(device)).cpu().numpy()
            prob_mask=sigmoid(outputs.astype(np.float64))[0][0]
            pred_mask = (prob_mask > 0.5).astype(int).tolist()
            prob_mask = prob_mask.tolist()
            preds+=pred_mask
            probs+=prob_mask
            testlabellist+=testlabel.numpy().tolist()
            testecoglist+=testecog.numpy().tolist()
            
        return preds, probs, testecoglist, testlabellist
    
    with st.form("my_form2", clear_on_submit=False):
        st.write("Seizure prediction using UNET1D on the whole data")
        
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            preds, probs, testecoglist, testlabellist = predict_whole_data(model, ecog_list, label_list)
            
            st.write(f"F1 Score: {f1_score(testlabellist, preds)}")
            st.write(f"Precision Score: {precision_score(testlabellist, preds)}")
            st.write(f"Recall Score: {recall_score(testlabellist, preds)}")

            st.write('Plotting ROC and PR Curve')
            fpr, tpr, thresholds = metrics.roc_curve(testlabellist, probs)
            precision, recall, _ = metrics.precision_recall_curve(testlabellist, probs)
            avg_precision = metrics.average_precision_score(testlabellist, probs)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

            # fig.suptitle(f'{filename}\n Avg. Precision: {avg_precision}')

            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                            estimator_name='unet model')
            display.plot(ax=ax1)

            precision, recall, _ = metrics.precision_recall_curve(testlabellist, probs)
            disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall,
                                                estimator_name='unet model', average_precision=avg_precision)
            # st.pyplot(disp.plot())
            # f, ax=plt.subplots(1, figsize=(20,8))
            disp.plot(ax=ax2)
            
            st.pyplot(fig)
            
            st.write('Plotting Confusion Matrix')
            # cm = confusion_matrix(testlabellist, preds, labels=[0,1])
            
            disp=metrics.ConfusionMatrixDisplay.from_predictions(testlabellist, preds,normalize='true',cmap='Blues')

            # disp.ax_.set_title(f"{filename}")
            #     disp.plot()
            # plt.show()
            fig, ax = plt.subplots(1,figsize=(20,8))
            disp.plot(ax=ax)
            st.pyplot(fig)
    
    @st.cache_data
    def download_mdic():
        preds, probs, testecoglist, testlabellist  = predict_whole_data(model, ecog_list, label_list)
        mdic={'predicted_label':preds, 'ecog':testecoglist, 'predicted_probs':probs,'actual_label':testlabellist}
        with open(f'predicted_{uploaded_file.name}', 'wb') as file:
            scipy.io.savemat(file, mdic, appendmat=True)
        # return scipy.io.savemat(f'predicted_{uploaded_file.name}', mdic, appendmat=True).encode('utf-8')

    # # mat_file = download_mdic()
    if st.button('Click to create prediction download file'):
        download_mdic()
        with open(f'predicted_{uploaded_file.name}', 'rb') as file:
            st.download_button(label="📥 Save the full prediction file",
                        data=file,
                        file_name=f'processed_{uploaded_file.name}',
                        #    mime="application/json"
                        )
    
    # with st.form("my_form3", clear_on_submit=False):  
    #     st.text_input("Filename (must include .mat)", key="filename")
    #     preds, probs, testecoglist, testlabellist = predict_whole_data(model, ecog_list, label_list)
    #     if 'preds' not in st.session_state:
    #         st.session_state['preds'] = preds
    #     if 'probs' not in st.session_state:
    #         st.session_state['probs'] = probs
    #     if 'testecoglist' not in st.session_state:
    #         st.session_state['testecoglist'] = testecoglist
    #     if 'testlabellist' not in st.session_state:
    #         st.session_state['testlabellist'] = testlabellist
    #     submit = st.form_submit_button("Download dataframe", on_click=download_mdic)
            
            
        # mdic={'predicted_label':preds, 'ecog':testecoglist, 'predicted_probs':probs,'actual_label':testlabellist}
        # st.download_button('Download full prediction file', mdic) 
                        


    