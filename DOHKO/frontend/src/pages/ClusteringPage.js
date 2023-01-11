import React, { useState, useEffect } from 'react';
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Code from '../components/Code';
import Console from '../components/Console';


export default function ClusteringPage({id=0}) {
	return (
		<Row className='justify-content-md-center' md={2}>
        <Col xs="auto" md={10}>

            <h4>Clustering Particional y Clasificación</h4>

            <Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Previsualización de datos</h5></Card.Title>
					<Console
						id_project={id}
						url={'clr/dataPreview/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                    <Card><Card.Body>
                    <Card.Title> Descripción de la estructura de datos</Card.Title>
                        
						<div className='mt-3'>
							<h6>Información de variables</h6>
                            <Console
                                id_project={id}
                                url={'clr/dataInfo/'}
                            />
                        </div>

						<div className='mt-3'>
							<h6>Agrupación por variable</h6>
                            <Console
                                id_project={id}
                                url={'clr/dataGroupBySize/'}
                            />
                        </div>
						
                    </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Eliminación de valores nulos</h5></Card.Title>
					<Console
						id_project={id}
						url={'clr/dataDropna/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Eliminación de variables</h5></Card.Title>
					<Console
						id_project={id}
						url={'clr/dataDropInplace/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Selección de características</h5></Card.Title>
					<Console
						id_project={id}
						url={'clr/dataCorrelation/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<h4 className='mt-4'>Segmentación Particional: K-means</h4>

			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				{/* <Card.Title>Segmentación Particional: K-means</Card.Title> */}
					
					<h6>Estandarización de datos</h6>
					<Console
						id_project={id}
						url={'clr/dataScaler/'}
					/>

					<h6>Definición de k clusters</h6>
					<Row>
						<Col>
							<Console
								id_project={id}
								url={'clr/dataClustersInit/'}
							/>
						</Col>
						<Col md={4}>
							<Console
								id_project={id}
								url={'clr/dataKneeLocator/'}
							/>
						</Col>
					</Row>

					<h6>Creación de etiquetas</h6>
					<Console
						id_project={id}
						url={'clr/dataClusterLebels/'}
					/>

					<h6>Obtención de centroides</h6>
					<Console
						id_project={id}
						url={'clr/dataGetGroups/'}
					/>

					{/* <h6>Gráfica de elementos y clusters</h6> */}
					
				</Card.Body></Card>
				</Col>
            </Row>

			<h4 className='mt-4'>Clasificación múltiple: Bosques Aleatorios</h4>

			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				{/* <Card.Title>Clasificación múltiple: Bosques Aleatorios</Card.Title> */}
					
					<h6>Variables predictoras</h6>
					<Console
						id_project={id}
						url={'rft/dataVPredict/'}
						update={true}
					/>
			
					<h6>Variable Clase</h6>
					<Console
						id_project={id}
						url={'rft/dataVPronostic/'}
						update={true}
					/>

					<h6>División de datos</h6>
					<Console
						id_project={id}
						url={'rft/dataDivision/'}
					/>

					<h6>Entrenamiento del modelo</h6>
					<Console
						id_project={id}
						url={'rft/dataModelTrainCRF/'}
					/>

					<div className='row mt-3'>
						<div className='col'>
							<h6>Medidas de rendimiento</h6>
							<Console
								id_project={id}
								url={'ptc/dataAccuracyScore/'}
							/>
							
							<h6 className='mt-3'>Matriz de clasificación</h6>
							<Console
								id_project={id}
								url={'ptc/dataMatrixClassification/'}
							/>
						</div>
						<div className='col'>
							<h6>Importancia de variables</h6>
							<Console
								id_project={id}
								url={'ptc/dataVariableStatus/'}
							/>
						</div>

					</div>

					{/* <h6>Clasificación final</h6>
					<Console
						id_project={id}
						url={'rft/dataClasification/'}
					/> */}

					<h6 className='mt-3'>Nuevas clasificaciones</h6>
					<Console
						id_project={id}
						url={'rft/dataNewPronostic/'}
						update={true}
					/>
					
				</Card.Body></Card>
				</Col>
            </Row>

		</Col>
		</Row>
	);
}