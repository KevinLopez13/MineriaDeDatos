import { useState } from 'react';
import FloatingLabel from 'react-bootstrap/FloatingLabel';
import Form from 'react-bootstrap/Form';

export default function Inputs({vals=[], input_vals, setInputVals}) {
    
    console.log(input_vals);

    const onChangeVal = (e) => {
        const updateVal = { [e.target.id] : e.target.value };
        setInputVals({...input_vals, ...updateVal});
    }


  return (
        <div style={{maxHeight:"350px", overflow:"auto", textAlign:"center"}}>
            <Form className='mx-2'>
            {vals.map((val)=>(
                <FloatingLabel className="mb-2" controlId={val.label} label={val.label}>
                
                {val.type === 'text' ?
                    <Form.Control type="text" defaultValue={val.default} onChange={(e)=>{onChangeVal(e)}} />
                    : null }

                {val.type === 'select' ?
                    <Form.Select aria-label="Floating label select example" onChange={(e)=>{onChangeVal(e)}}>
                        {val.options.map((option) => (
                            <option value={option.value}>{option.label}</option>
                        ))}
                    </Form.Select> : null }
                </FloatingLabel>
            ))}
            </Form>
        </div>
  );
}