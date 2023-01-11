import React, { useState, useEffect } from 'react';
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Console from '../components/Console';


export default function ClasificationPage({id=0}) {
	return (
		<Row className='justify-content-md-center' md={2}>
        <Col xs="auto" md={10}>

            <h4>Clasificación</h4>

            <Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Previsualización de datos</h5></Card.Title>
					<Console
						id_project={id}
						url={'rft/dataPreview/'}
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
                                url={'rft/dataInfo/'}
                            />
                        </div>

                        <div className='mt-3'>
							<h6>Resumen estadístico de variables numéricas</h6>
							<Console
								id_project={id}
								url={'rft/dataDescribe/'}
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
						url={'rft/dataDropna/'}
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

			{/* Arboles de Decisión */}
			
			<h4 className='mt-4'>Árboles de Decisión</h4>
			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				{/* <Card.Title>Árboles de Decisión</Card.Title> */}
					
					<h6 className='mt-3'>Generación del pronóstico</h6>
					<Console
						id_project={id}
						url={'ptc/dataModelTrainCDT/'}
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

					<h6 className='mt-3'>Conformación del árbol</h6>
					<Console
						id_project={id}
						url={'rft/dataPlotClasTree/'}
					/>
					
					<h6 className='mt-3'>Nuevas clasificaciones</h6>
					<Console
						id_project={id}
						url={'ptc/dataNewPronostic/'}
						update={true}
					/>
				</Card.Body></Card>
				</Col>
            </Row>


			<h4 className='mt-4'>Bosques Aleatorios</h4>
			<Row className='mt-4'>
				<Col>
				<Card><Card.Body>
				{/* <Card.Title>Bosques Aleatorios</Card.Title> */}

					<h6 className='mt-3'>Entrenamiento del modelo</h6>
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

					{/* <h6 className='mt-3'>Conformación del árbol</h6>
					<Console
						id_project={id}
						url={'rft/dataPlotDTree/'}
					/> */}
					
					<h6 className='mt-3'>Nuevos pronósticos</h6>
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