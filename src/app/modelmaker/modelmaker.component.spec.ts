import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelmakerComponent } from './modelmaker.component';

describe('ModelmakerComponent', () => {
  let component: ModelmakerComponent;
  let fixture: ComponentFixture<ModelmakerComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ModelmakerComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ModelmakerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
