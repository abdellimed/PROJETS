
import joblib

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from fastapi import FastAPI
from pydantic import BaseModel
import string
import gradio as gr
from elasticsearch import Elasticsearch
from datetime import datetime
import pytz
import asyncio
import os

app = FastAPI()

class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) et pas les poids du modèle
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def training_step(self, batch):
        out = self.forward(batch)

        logits = out.logits
        # -------- MASKED --------
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )




    
type_lable={'Objets abandonnés': 0,
 'Graffitis, tags, affiches et autocollants': 1,
 'Autos, motos, vélos, trottinettes...': 2,
 'Activités commerciales et professionnelles': 3,
 'Arbres, végétaux et animaux': 4,
 'Éclairage / Électricité': 5,
 'Propreté': 6,
 'Mobiliers urbains': 7,
 'Voirie et espace public': 8,
 'Eau': 9}


ID_TO_LABEL=list(type_lable.keys())
tokenizer = AutoTokenizer.from_pretrained('camembert-base')
camembert = CamembertForMaskedLM.from_pretrained('camembert-base')
lightning_model = LightningModel.load_from_checkpoint(checkpoint_path="../models/version_1/checkpoints/epoch=2-step=8085.ckpt")

elasticsearch_uri =os.environ.get('SERVER_URI_ELASTICSEARCH')
es=Elasticsearch([elasticsearch_uri])
france_zone=pytz.timezone('Europe/Paris')

class Commentaire(BaseModel):
    Comment: str
@app.get("/")
def root():
    return {"API pour classification des commentaires" }



@app.post("/Comment-class")
def post_Comment_class(clsT: Commentaire):
    device=torch.device('cpu')
    Model=lightning_model.model
    label_predicted, proba = get_preds(Model.to(device), tokenizer, clsT.Comment)
    Comment_id = datetime.now(france_zone)
    es.index(index='verbatim_surveillance', body={'date':Comment_id,'Commentaire': clsT.Comment,'label': label_predicted ,'score':proba})

    return {'label':label_predicted, 'score':proba}
    
def get_preds(model, tokenizer, sentence):
    type_lable={'Objets abandonnés': 0,
     'Graffitis, tags, affiches et autocollants': 1,
     'Autos, motos, vélos, trottinettes...': 2,
     'Activités commerciales et professionnelles': 3,
     'Arbres, végétaux et animaux': 4,
     'Éclairage / Électricité': 5,
     'Propreté': 6,
     'Mobiliers urbains': 7,
     'Voirie et espace public': 8,
     'Eau': 9}


    ID_TO_LABEL=list(type_lable.keys())
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    input_ids, attention_mask = tokenized_sentence.input_ids, tokenized_sentence.attention_mask

    out = model(
        input_ids=tokenized_sentence.input_ids,
        attention_mask=tokenized_sentence.attention_mask
    )

    logits = out.logits

    probas = torch.softmax(logits, -1).squeeze()

    pred = torch.argmax(probas)

    return ID_TO_LABEL[pred], probas[pred].item()
def get_prediction(comment):
    device=torch.device('cpu')
    Model=lightning_model.model
    label_predicted, proba = get_preds(Model.to(device), tokenizer, comment)
    Comment_id = datetime.now(france_zone)
    es.index(index='verbatim_g_surveillance', body={'date':Comment_id,'Commentaire': comment,'label': label_predicted ,'score':proba})

    return label_predicted, proba

gr_interface=gr.Interface(
    fn=get_prediction,
    inputs=gr.inputs.Textbox(label='Entrez le commentaire ici',lines=2),
    outputs=[gr.outputs.Textbox(label="Label: "),gr.outputs.Textbox(label="Score de confiance :")],
    title="DMR - Classification",
    allow_flagging=False
)

app=gr.mount_gradio_app(app, gr_interface,path="/classification")