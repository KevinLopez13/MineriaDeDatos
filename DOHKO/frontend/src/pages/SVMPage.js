import React from 'react';
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Console from '../components/Console';


export default function SVMPage({id=0}) {
	return (
		<Row className='justify-content-md-center' md={2}>
        <Col xs="auto" md={10}>

            <h4>Máquinas de Vectores de Soporte</h4>

            <Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Previsualización de datos</h5></Card.Title>
					<Console
						id_project={id}
						url={'svm/dataPreview/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                    <Card><Card.Body>
                    <Card.Title> Descripción de la estructura de datos</Card.Title>
						
						<div className='mt-3'>
							<h6>Agrupación por variable</h6>
                            <Console
                                id_project={id}
                                url={'clr/dataGroupBySize/'}
                            />
                        </div>
						
                        <div className='mt-3'>
							<h6>Resumen estadístico de variables numéricas</h6>
							<Console
								id_project={id}
								url={'svm/dataDescribe/'}
							/>
                        </div>
                    </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Selección de características</h5></Card.Title>
					<Console
						id_project={id}
						url={'svm/dataCorrelation/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				<Card.Title>Definición de las variables</Card.Title>
					
					<h6>Variables predictoras</h6>
					<Console
						id_project={id}
						url={'svm/dataVPredict/'}
					/>
			
					<h6>Variable a clase</h6>
					<Console
						id_project={id}
						url={'svm/dataVPronostic/'}
					/>

					<h6>División de datos</h6>
					<Console
						id_project={id}
						url={'svm/dataDivision/'}
					/>
				</Card.Body></Card>
				</Col>
            </Row>

			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				<Card.Title>Máquina de soporte vactorial (SVM)</Card.Title>
					
					<h6 className='mt-3'>Entrenamiento del modelo</h6>
					<Console
						id_project={id}
						url={'svm/dataModelTrainSVM/'}
					/>

					<div className='row mt-3'>
						<div className='col'>
							<h6>Medida de rendimiento</h6>
							<Console
								id_project={id}
								url={'ptc/dataMeanScore/'}
							/>
							
							<h6 className='mt-3'>Matriz de clasificación</h6>
							<Console
								id_project={id}
								url={'ptc/dataMatrixClassification/'}
							/>
						</div>

						<div className='col'>
							<h6>Curva ROC</h6>
							<Console
								id_project={id}
								url={'ptc/dataPlotROC/'}
							/>
						</div>

					</div>

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