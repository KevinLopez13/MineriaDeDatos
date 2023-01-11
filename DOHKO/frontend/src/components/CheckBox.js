import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';


export default function CheckBox({data=[], type, vars, setVars}) {

    if (data === null){
        return null;
    }

    const handle = (e) => {
        
        if (type === 'checkbox'){

            if ( e.target.checked ) {
                setVars([...vars, e.target.id]);
            }
            else{
                const newVars = vars.filter( v => v !== e.target.id );
                setVars(newVars);
            }
        }
        
        else{
            setVars([e.target.id]);
        }
    };
    

    return (
        <div style={{maxHeight:"350px", overflow:"auto"}} >
            {data.map((variable) => (
                <div className="mx-2 my-2">
                    <Form.Check
                        onChange={(e)=>{handle(e)}}
                        name="vars"
                        type={type}
                        id={variable}
                        label={variable}
                        style={{fontSize:14}}
                    />
                </div>
            ))}
        </div>
    )
}