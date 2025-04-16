# database/schemas.py
from sqlalchemy import create_engine, Column, Integer, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = 'radiosensitivity'
    
    id = Column(Integer, primary_key=True)
    telomere_lengths = Column(ARRAY(Float))  # [baseline, 24h, 72h, 10d]
    clinical_params = Column(ARRAY(Float))   # [age, tumor_volume]
    attention_weights = Column(ARRAY(Float)) # Model interpretation
    predicted_risk = Column(Float)
